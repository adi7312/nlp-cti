"""BERT-BiLSTM-CRF Automatic Entity and Relation Extraction.

This module implements a deep learning-based approach for named entity recognition
(NER) and relation extraction using BERT embeddings, BiLSTM layers, and CRF decoding.

When trained models are unavailable, falls back to:
- HuggingFace pre-trained NER pipeline (SecureModernBERT-NER) for entity extraction
- Local LLM for relation extraction
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup, pipeline as hf_pipeline
from torch.optim import AdamW
from torchcrf import CRF
import re
import json

class NERDataset(Dataset):
    """Dataset for Named Entity Recognition."""

    def __init__(self, texts: List[List[str]], labels: List[List[str]], tokenizer: BertTokenizer, label2idx: Dict[str, int], max_len: int = 128) -> None:
        """Initialize NER dataset.

        Args:
            texts: List of tokenized text sequences.
            labels: List of corresponding label sequences.
            tokenizer: BERT tokenizer instance.
            label2idx: Dictionary mapping label names to indices.
            max_len: Maximum sequence length (default: 128).
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            Dictionary containing input_ids, attention_mask, and labels.
        """
        text = self.texts[idx]
        word_labels = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            is_split_into_words=True
        )

        labels = []
        word_ids = encoding.word_ids(batch_index=0)
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD])
                labels.append(self.label2idx.get('O', 0))
            elif word_idx != previous_word_idx:
                # First sub-token of a word
                label = word_labels[word_idx]
                labels.append(self.label2idx.get(label, self.label2idx.get('O', 0)))
            else:
                # Subsequent sub-tokens of a word
                label = word_labels[word_idx]
                # Option: use 'I-' version of the label or 'O' or keep the same
                # Standard practice for NER with BERT: use the same label or a special 'X' label
                # Here we use the same label but could also use I- version if it was B-
                if label.startswith('B-'):
                    label = 'I-' + label[2:]
                labels.append(self.label2idx.get(label, self.label2idx.get('O', 0)))
            previous_word_idx = word_idx

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

class RelationDataset(Dataset):
    """Dataset for Relation Extraction."""

    def __init__(self, texts: List[List[str]], entities: List[List[Tuple[int, int, str]]], relations: List[List[Tuple[int, int, str]]],
                tokenizer: BertTokenizer, rel2idx: Dict[str, int], max_len: int = 128, max_entities: int = 32) -> None:
        """Initialize relation extraction dataset.

        Args:
            texts: List of tokenized text sequences.
            entities: List of entity spans (start, end, type) for each text.
            relations: List of relation triples (head_idx, tail_idx, rel_type) for each text.
            tokenizer: BERT tokenizer instance.
            rel2idx: Dictionary mapping relation names to indices.
            max_len: Maximum sequence length (default: 128).
            max_entities: Maximum number of entities per sequence (default: 32).
        """
        self.texts = texts
        self.entities = entities
        self.relations = relations
        self.tokenizer = tokenizer
        self.rel2idx = rel2idx
        self.max_len = max_len
        self.max_entities = max_entities

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            Dictionary containing input_ids, attention_mask, entity_positions, and relation_matrix.
        """
        text = self.texts[idx]
        entity_spans = self.entities[idx][:self.max_entities]
        relation_triples = self.relations[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            is_split_into_words=True
        )

        word_ids = encoding.word_ids(batch_index=0)

        # Map word indices to sub-token indices
        word_to_subtokens = {}
        for sub_idx, word_idx in enumerate(word_ids):
            if word_idx is not None:
                if word_idx not in word_to_subtokens:
                    word_to_subtokens[word_idx] = []
                word_to_subtokens[word_idx].append(sub_idx)

        # Create entity position matrix (max_entities x seq_len)
        entity_positions = torch.zeros((self.max_entities, self.max_len))

        for e_idx, (start, end, _) in enumerate(entity_spans):
            # start, end are indices in the original 'text' list
            subtoken_indices = []
            for word_idx in range(start, end):
                if word_idx in word_to_subtokens:
                    subtoken_indices.extend(word_to_subtokens[word_idx])

            for sub_idx in subtoken_indices:
                if sub_idx < self.max_len:
                    entity_positions[e_idx, sub_idx] = 1

        # Create relation matrix (max_entities x max_entities)
        # We use a single label per pair for CrossEntropyLoss
        relation_matrix = torch.zeros((self.max_entities, self.max_entities), dtype=torch.long)
        # Initialize with 'no_relation' index (usually 0)
        no_rel_idx = self.rel2idx.get('no_relation', 0)
        relation_matrix.fill_(no_rel_idx)

        num_actual_entities = len(entity_spans)
        for head_idx, tail_idx, rel_type in relation_triples:
            if head_idx < num_actual_entities and tail_idx < num_actual_entities:
                rel_idx = self.rel2idx.get(rel_type, no_rel_idx)
                relation_matrix[head_idx, tail_idx] = rel_idx

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'entity_positions': entity_positions,
            'labels': relation_matrix
        }

class BERTBiLSTMCRF(nn.Module):
    """BERT-BiLSTM-CRF model for Named Entity Recognition.

    Architecture:
        BERT -> BiLSTM -> CRF
    """

    def __init__(self, bert_model_name: str = 'bert-base-uncased', hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1,
                label2idx: Optional[Dict[str, int]] = None) -> None:
        """Initialize BERT-BiLSTM-CRF model.

        Args:
            bert_model_name: Name of the BERT model to use (default: 'bert-base-uncased').
            hidden_size: Size of BiLSTM hidden state (default: 128).
            num_layers: Number of BiLSTM layers (default: 2).
            dropout: Dropout probability (default: 0.1).
            label2idx: Dictionary mapping label names to indices.
        """
        super(BERTBiLSTMCRF, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size

        self.bilstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        self.label2idx = label2idx or {'O': 0}
        self.num_labels = len(self.label2idx)

        # Linear layer to project BiLSTM output to label space
        self.classifier = nn.Linear(hidden_size * 2, self.num_labels)

        # CRF layer
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the model.

        Args:
            input_ids: Input token IDs (batch_size, seq_len).
            attention_mask: Attention mask (batch_size, seq_len).
            labels: Optional ground truth labels (batch_size, seq_len).

        Returns:
            Tuple of (logits or emissions, loss). Loss is None if labels not provided.
        """
        # BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state

        # BiLSTM
        lstm_output, _ = self.bilstm(sequence_output)
        lstm_output = self.dropout(lstm_output)

        # Classification layer
        emissions = self.classifier(lstm_output)

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool())
            return emissions, loss
        else:
            return emissions, None

    def decode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[List[str]]:
        """Decode predictions using CRF.

        Args:
            input_ids: Input token IDs (batch_size, seq_len).
            attention_mask: Attention mask (batch_size, seq_len).

        Returns:
            List of predicted label sequences (batch_size, seq_len).
        """
        idx2label = {v: k for k, v in self.label2idx.items()}

        emissions, _ = self.forward(input_ids, attention_mask)

        batch_predictions = []
        for emission, mask in zip(emissions, attention_mask.bool()):
            # Decode using CRF
            tags = self.crf.decode(emission.unsqueeze(0), mask=mask.unsqueeze(0))[0]

            # Convert indices to label names
            predictions = [idx2label.get(tag, 'O') for tag in tags]
            batch_predictions.append(predictions)

        return batch_predictions

class BERTBiLSTMRelationModel(nn.Module):
    """BERT-BiLSTM model for Relation Extraction.

    Architecture:
        BERT -> BiLSTM -> Entity-aware attention -> Relation classification
    """

    def __init__(self, bert_model_name: str = 'bert-base-uncased', hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1,
                rel2idx: Optional[Dict[str, int]] = None, max_entities: int = 32) -> None:
        """Initialize BERT-BiLSTM relation extraction model.

        Args:
            bert_model_name: Name of the BERT model to use (default: 'bert-base-uncased').
            hidden_size: Size of BiLSTM hidden state (default: 128).
            num_layers: Number of BiLSTM layers (default: 2).
            dropout: Dropout probability (default: 0.1).
            rel2idx: Dictionary mapping relation names to indices.
            max_entities: Maximum number of entities per document (default: 32).
        """
        super(BERTBiLSTMRelationModel, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size

        self.bilstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        self.rel2idx = rel2idx or {'no_relation': 0}
        self.num_relations = len(self.rel2idx)
        self.max_entities = max_entities

        # Entity-aware attention
        self.entity_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            batch_first=True
        )

        # Relation classification
        self.relation_classifier = nn.Linear(hidden_size * 2 * 2, self.num_relations)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, entity_positions: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the model.

        Args:
            input_ids: Input token IDs (batch_size, seq_len).
            attention_mask: Attention mask (batch_size, seq_len).
            entity_positions: Entity position matrix (batch_size, num_entities, seq_len).
            labels: Optional ground truth relation indices (batch_size, num_entities, num_entities).

        Returns:
            Tuple of (relation_logits, loss). Loss is None if labels not provided.
        """
        # BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state

        # BiLSTM
        lstm_output, _ = self.bilstm(sequence_output)
        lstm_output = self.dropout(lstm_output)

        # Get entity representations using entity positions
        # Vectorized version of weighted average pooling
        # entity_positions: (batch_size, num_entities, seq_len)
        # lstm_output: (batch_size, seq_len, hidden_size * 2)
        entity_sums = torch.matmul(entity_positions, lstm_output) # (batch_size, num_entities, hidden_size * 2)
        entity_lengths = entity_positions.sum(dim=2, keepdim=True) # (batch_size, num_entities, 1)
        entity_reps = entity_sums / (entity_lengths + 1e-8)

        # Entity-aware attention
        # Mask for padded entities (where length is 0)
        # key_padding_mask: (batch_size, num_entities) where True means padded
        entity_mask = (entity_lengths.squeeze(-1) == 0)

        entity_reps_attn, _ = self.entity_attention(
            entity_reps, entity_reps, entity_reps,
            key_padding_mask=entity_mask
        )
        entity_reps = entity_reps + entity_reps_attn

        # Pair entities for relation classification
        num_entities = entity_reps.size(1)
        head_reps = entity_reps.unsqueeze(2).expand(-1, -1, num_entities, -1)
        tail_reps = entity_reps.unsqueeze(1).expand(-1, num_entities, -1, -1)

        pair_reps = torch.cat([head_reps, tail_reps], dim=-1) # (batch_size, num_entities, num_entities, hidden_size * 4)

        # Relation classification
        relation_logits = self.relation_classifier(pair_reps) # (batch_size, num_entities, num_entities, num_relations)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Flatten only the first 3 dimensions for the batch/pairs
            loss = loss_fct(
                relation_logits.view(-1, self.num_relations),
                labels.view(-1)
            )
            return relation_logits, loss

        return relation_logits, None


class EntityRelationExtractor:
    """Main class for entity and relation extraction.

    Provides a unified interface for training and inference using BERT-BiLSTM-CRF
    models for both NER and relation extraction.
    """

    def __init__(self, bert_model_name: str = 'bert-base-uncased', ner_hidden_size: int = 128, rel_hidden_size: int = 128,
                num_layers: int = 2, dropout: float = 0.1, max_entities: int = 32, device: Optional[str] = None,
                llm: Any = None) -> None:
        """Initialize entity and relation extractor.

        Args:
            bert_model_name: Name of the BERT model to use (default: 'bert-base-uncased').
            ner_hidden_size: Hidden size for NER BiLSTM (default: 128).
            rel_hidden_size: Hidden size for relation BiLSTM (default: 128).
            num_layers: Number of BiLSTM layers (default: 2).
            dropout: Dropout probability (default: 0.1).
            max_entities: Maximum number of entities per sequence (default: 32).
            device: Device for model training/inference. Uses CUDA if available.
            llm: Optional LangChain chat model for LLM-based relation extraction fallback.
        """
        self.bert_model_name = bert_model_name
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.ner_model: Optional[BERTBiLSTMCRF] = None
        self.rel_model: Optional[BERTBiLSTMRelationModel] = None

        self.ner_hidden_size = ner_hidden_size
        self.rel_hidden_size = rel_hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_entities = max_entities

        self.label2idx = {'O': 0}
        self.rel2idx = {'no_relation': 0}

        self.hf_ner = hf_pipeline(
            "token-classification",
            model="attack-vector/SecureModernBERT-NER",
            aggregation_strategy="first"
        )
        self.llm = llm
        self._re_attempts = 0
        self._re_successes = 0

    def fit_ner(self, train_texts: List[List[str]], train_labels: List[List[str]], label2idx: Dict[str, int], batch_size: int = 16,
                epochs: int = 10, learning_rate: float = 2e-5, warmup_steps: int = 100) -> BERTBiLSTMCRF:
        """Train the NER model.

        Args:
            train_texts: List of tokenized text sequences.
            train_labels: List of corresponding label sequences.
            label2idx: Dictionary mapping label names to indices.
            batch_size: Batch size for training (default: 16).
            epochs: Number of training epochs (default: 10).
            learning_rate: Learning rate for optimizer (default: 2e-5).
            warmup_steps: Number of warmup steps for scheduler (default: 100).

        Returns:
            Trained BERT-BiLSTM-CRF model.
        """
        # Create label mapping
        self.label2idx = label2idx
        self.idx2label = {v: k for k, v in label2idx.items()}

        # Create dataset and dataloader
        dataset = NERDataset(
            texts=train_texts,
            labels=train_labels,
            tokenizer=self.tokenizer,
            label2idx=label2idx
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # Initialize model
        self._init_ner_model()

        # Optimizer and scheduler
        optimizer = AdamW(self.ner_model.parameters(), lr=learning_rate)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Training loop
        self.ner_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                _, loss = self.ner_model(input_ids, attention_mask, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"NER Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        return self.ner_model

    def predict_entities(self, texts: List[List[str]]) -> List[List[Tuple[int, int, str]]]:
        """Predict entities in text sequences.

        Args:
            texts: List of tokenized text sequences.

        Returns:
            List of entity spans (start, end, type) for each text.
        """
        if self.ner_model is None:
            raise ValueError("NER model not trained. Call fit_ner() first.")

        self.ner_model.eval()
        # Use dummy labels for NERDataset
        dataset = NERDataset(
            texts=texts,
            labels=[['O'] * len(t) for t in texts],
            tokenizer=self.tokenizer,
            label2idx=self.label2idx
        )
        dataloader = DataLoader(dataset, batch_size=16)

        all_entities = []
        batch_start = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                predictions = self.ner_model.decode(input_ids, attention_mask)

                for i, preds in enumerate(predictions):
                    # Map sub-token predictions back to original words
                    original_text = texts[batch_start + i]
                    encoding = self.tokenizer(
                        original_text,
                        is_split_into_words=True
                    )
                    word_ids = encoding.word_ids()

                    word_labels = []
                    previous_word_idx = None
                    for sub_idx, word_idx in enumerate(word_ids):
                        if word_idx is not None and word_idx != previous_word_idx:
                            if sub_idx < len(preds):
                                word_labels.append(preds[sub_idx])
                        previous_word_idx = word_idx

                    entities = self._extract_entities_from_predictions(word_labels)
                    all_entities.append(entities)

                batch_start += input_ids.size(0)

        return all_entities

    def _extract_entities_from_predictions(self, predictions: List[str]) -> List[Tuple[int, int, str]]:
        """Extract entity spans from predicted labels.

        Args:
            predictions: List of predicted labels.

        Returns:
            List of entity spans (start, end, type).
        """
        entities = []
        current_entity = None

        for i, label in enumerate(predictions):
            if label.startswith('B-'):
                if current_entity is not None:
                    entities.append(current_entity)
                entity_type = label[2:]
                current_entity = (i, i + 1, entity_type)
            elif label.startswith('I-'):
                if current_entity is not None:
                    entity_type = label[2:]
                    start, _, et = current_entity
                    if et == entity_type:
                        current_entity = (start, i + 1, entity_type)
                    else:
                        entities.append(current_entity)
                        current_entity = (i, i + 1, entity_type)
                else:
                    # I- without B-, treat as B-
                    current_entity = (i, i + 1, label[2:])
            else:
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = None

        if current_entity is not None:
            entities.append(current_entity)

        return entities

    def fit_relation(self, train_texts: List[List[str]], train_entities: List[List[Tuple[int, int, str]]], train_relations: List[List[Tuple[int, int, str]]],
                    rel2idx: Dict[str, int], batch_size: int = 8, epochs: int = 15, learning_rate: float = 2e-5, warmup_steps: int = 100) -> BERTBiLSTMRelationModel:
        """Train the relation extraction model.

        Args:
            train_texts: List of tokenized text sequences.
            train_entities: List of entity spans for each text.
            train_relations: List of relation triples for each text.
            rel2idx: Dictionary mapping relation names to indices.
            batch_size: Batch size for training (default: 8).
            epochs: Number of training epochs (default: 15).
            learning_rate: Learning rate for optimizer (default: 2e-5).
            warmup_steps: Number of warmup steps for scheduler (default: 100).

        Returns:
            Trained BERT-BiLSTM relation extraction model.
        """
        self.rel2idx = rel2idx
        self.idx2rel = {v: k for k, v in rel2idx.items()}

        dataset = RelationDataset(
            texts=train_texts,
            entities=train_entities,
            relations=train_relations,
            tokenizer=self.tokenizer,
            rel2idx=rel2idx,
            max_entities=self.max_entities
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self._init_rel_model()

        optimizer = AdamW(self.rel_model.parameters(), lr=learning_rate)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        self.rel_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                entity_positions = batch['entity_positions'].to(self.device)
                relation_matrix = batch['labels'].to(self.device)

                optimizer.zero_grad()
                _, loss = self.rel_model(
                    input_ids, attention_mask, entity_positions, relation_matrix
                )
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Relation Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        return self.rel_model

    def predict_relations(self, texts: List[List[str]], entities: List[List[Tuple[int, int, str]]]) -> List[List[Tuple[int, int, str]]]:
        """Predict relations in text sequences given entities.

        Args:
            texts: List of tokenized text sequences.
            entities: List of entity spans for each text.

        Returns:
            List of relation triples (head_idx, tail_idx, relation_type) for each text.
        """
        if self.rel_model is None:
            raise ValueError("Relation model not trained. Call fit_relation() first.")

        self.rel_model.eval()
        dataset = RelationDataset(
            texts=texts,
            entities=entities,
            relations=[[] for _ in texts],  # Dummy relations
            tokenizer=self.tokenizer,
            rel2idx=self.rel2idx,
            max_entities=self.max_entities
        )
        dataloader = DataLoader(dataset, batch_size=8)

        all_relations = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                entity_positions = batch['entity_positions'].to(self.device)

                relation_logits, _ = self.rel_model(
                    input_ids, attention_mask, entity_positions
                )

                for rel_logits in relation_logits:
                    relations = self._extract_relations_from_logits(rel_logits)
                    all_relations.append(relations)

        return all_relations

    def _extract_relations_from_logits(self, relation_logits: torch.Tensor) -> List[Tuple[int, int, str]]:
        """Extract relations from model logits.

        Args:
            relation_logits: Relation logits (num_entities, num_entities, num_relations).

        Returns:
            List of relation triples (head_idx, tail_idx, relation_type).
        """
        num_entities = relation_logits.size(0)
        relations = []

        for head_idx in range(num_entities):
            for tail_idx in range(num_entities):
                if head_idx == tail_idx:
                    continue

                rel_probs = torch.softmax(relation_logits[head_idx, tail_idx], dim=0)
                pred_rel_idx = torch.argmax(rel_probs).item()

                if pred_rel_idx != self.rel2idx.get('no_relation', 0):
                    relation_type = self.idx2rel.get(pred_rel_idx, 'unknown')
                    relations.append((head_idx, tail_idx, relation_type))

        return relations

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract entities and relations from raw text.

        Uses BERT-BiLSTM-CRF if trained models are available.
        Falls back to HuggingFace pre-trained NER + optional LLM for RE otherwise.

        Args:
            text: Raw input text.

        Returns:
            Dictionary containing extracted entities and relations.
        """
        # This regex matches words, numbers, and punctuation marks as separate tokens
        tokens = re.findall(r"[\w']+|[.,!?;:\"()\[\]\-]", text)

        if self.ner_model:
            entities = self.predict_entities([tokens])[0]
        else:
            entities = self._extract_entities_hf(text)

        if self.rel_model and entities:
            relations = self.predict_relations([tokens], [entities])[0]
        elif self.llm and entities:
            relations = self._extract_relations_with_llm(text, tokens, entities)
        else:
            if not entities:
                pass  # no entities to relate
            elif not self.llm:
                print("  [EXTRACT] ⚠ entities found but llm is None — RE disabled")
            relations = []

        return {
            'text': text,
            'tokens': tokens,
            'entities': [
                {'start': start, 'end': end, 'type': etype}
                for start, end, etype in entities
            ],
            'relations': [
                {'head': head, 'tail': tail, 'type': rtype}
                for head, tail, rtype in relations
            ]
        }

    def _extract_entities_hf(self, text: str) -> List[Tuple[int, int, str]]:
        """Extract entities using HuggingFace pre-trained NER pipeline.

        Maps character-level spans from the HF pipeline to token-level spans
        compatible with the existing entity format.

        Args:
            text: Raw input text.

        Returns:
            List of (token_start, token_end, entity_type) tuples.
        """
        matches = list(re.finditer(r"[\w']+|[.,!?;:\"()\[\]\-]", text))
        tokens = [m.group() for m in matches]

        char_to_token = {}
        for idx, m in enumerate(matches):
            for c in range(m.start(), m.end()):
                char_to_token[c] = idx

        hf_entities = self.hf_ner(text)

        seen = set()
        entities = []
        for ent in hf_entities:
            etype = ent.get('entity_group', ent.get('entity', 'UNKNOWN'))
            if etype.startswith('B-') or etype.startswith('I-'):
                etype = etype[2:]
            start_char = ent['start']
            end_char = ent['end']

            # HF spans often include leading whitespace or trailing punctuation
            # that our regex tokenizer skips. Walk to the nearest mapped char.
            token_start = char_to_token.get(start_char)
            if token_start is None:
                for c in range(start_char + 1, min(start_char + 20, len(text))):
                    if c in char_to_token:
                        token_start = char_to_token[c]
                        break
            token_end = None
            for c in range(end_char - 1, start_char - 1, -1):
                if c in char_to_token:
                    token_end = char_to_token[c] + 1
                    break
            # Also try walking forward a bit from end_char
            if token_end is None:
                for c in range(end_char, min(end_char + 10, len(text))):
                    if c in char_to_token:
                        token_end = char_to_token[c] + 1
                        break

            if token_start is not None and token_end is not None:
                key = (token_start, token_end, etype)
                if key not in seen:
                    seen.add(key)
                    entities.append((token_start, token_end, etype))

        return entities

    def _extract_relations_with_llm(
        self, text: str, tokens: List[str], entities: List[Tuple[int, int, str]]
    ) -> List[Tuple[int, int, str]]:
        """Extract relations between entities using the local LLM.

        Args:
            text: Raw input text.
            tokens: Tokenized text.
            entities: List of (token_start, token_end, entity_type) tuples.

        Returns:
            List of (head_idx, tail_idx, relation_type) tuples.
        """
        if len(entities) < 2:
            return []

        self._re_attempts += 1

        name_to_idx = {}
        entity_lines = []
        for i, (start, end, etype) in enumerate(entities):
            name = " ".join(tokens[start:end])
            entity_lines.append(f"  {name} ({etype})")
            name_to_idx[name] = i

        entities_str = "\n".join(entity_lines)

        # Truncate text to prevent blowing the LLM context window
        text_truncated = text[:2000] if len(text) > 2000 else text

        prompt = (
            "Extract relationships between named entities in this CTI text.\n\n"
            f"--- Entities ---\n{entities_str}\n\n"
            f"--- Text ---\n{text_truncated}\n\n"
            "--- Task ---\n"
            "Return ONLY a JSON array. Each element must be:\n"
            '  {"head": "<entity name>", "tail": "<entity name>", "relation": "<TYPE>"}\n'
            'Example: [{"head": "APT29", "tail": "CVE-2021-26855", "relation": "USES"}]\n\n'
            "Use these relation types: TARGETS, USES, ATTRIBUTED_TO, CONNECTS_TO, ASSOCIATED_WITH, EXPLOITS, HAS_TYPE, LOCATED_IN\n"
            "Return [] if no clear relationships exist.\n"
            "JSON array:"
        )

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()

            # Strip markdown fences if present
            if "```" in content:
                content = content.split("```")[-2] if content.count("```") >= 2 else content.split("```")[-1]
                content = content.strip()
            # Some models prepend "json" before the JSON payload
            content = content.removeprefix("json").removeprefix("JSON").strip()

            if not content:
                print("  [RE] Empty content after stripping prefixes")
                return []

            relations_data = json.loads(content)
            if not isinstance(relations_data, list):
                print(f"  [RE] LLM returned non-list JSON: {type(relations_data).__name__}")
                return []

            relations = []
            for rel in relations_data:
                head_name = rel.get("head", "")
                tail_name = rel.get("tail", "")
                if not head_name or not tail_name:
                    continue
                rtype = rel.get("relation", "RELATED_TO").upper().replace(" ", "_")

                head_idx = name_to_idx.get(head_name)
                tail_idx = name_to_idx.get(tail_name)

                if head_idx is not None and tail_idx is not None and head_idx != tail_idx:
                    relations.append((head_idx, tail_idx, rtype))

            if relations:
                self._re_successes += 1
            return relations

        except json.JSONDecodeError as e:
            print(f"  [RE] JSON parse error: {e}")
            print(f"  [RE] Raw LLM response: {content[:300]}")
            return []
        except Exception as e:
            print(f"  [RE] LLM call failed: {type(e).__name__}: {e}")
            return []

    def _init_ner_model(self) -> None:
        """Initialize NER model."""
        self.ner_model = BERTBiLSTMCRF(
            bert_model_name=self.bert_model_name,
            hidden_size=self.ner_hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            label2idx=self.label2idx
        ).to(self.device)

    def _init_rel_model(self) -> None:
        """Initialize relation extraction model."""
        self.rel_model = BERTBiLSTMRelationModel(
            bert_model_name=self.bert_model_name,
            hidden_size=self.rel_hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            rel2idx=self.rel2idx,
            max_entities=self.max_entities
        ).to(self.device)

    def save(self, save_path: str) -> None:
        """Save models and configuration to disk.

        Args:
            save_path: Directory to save models.
        """
        import os
        import json
        os.makedirs(save_path, exist_ok=True)

        # Save configuration
        config = {
            'bert_model_name': self.bert_model_name,
            'ner_hidden_size': self.ner_hidden_size,
            'rel_hidden_size': self.rel_hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'max_entities': self.max_entities,
            'label2idx': self.label2idx,
            'rel2idx': self.rel2idx
        }
        with open(f"{save_path}/config.json", 'w') as f:
            json.dump(config, f, indent=4)

        if self.ner_model:
            torch.save(self.ner_model.state_dict(), f"{save_path}/ner_model.pt")
        if self.rel_model:
            torch.save(self.rel_model.state_dict(), f"{save_path}/rel_model.pt")

        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)

    def load(self, load_path: str) -> None:
        """Load models and configuration from disk.

        Args:
            load_path: Directory containing saved models.
        """
        import json
        import os

        # Load configuration
        with open(f"{load_path}/config.json", 'r') as f:
            config = json.load(f)

        self.bert_model_name = config['bert_model_name']
        self.ner_hidden_size = config['ner_hidden_size']
        self.rel_hidden_size = config['rel_hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.max_entities = config['max_entities']
        self.label2idx = config['label2idx']
        self.rel2idx = config['rel2idx']

        self.idx2label = {int(v): k for k, v in self.label2idx.items()}
        self.idx2rel = {int(v): k for k, v in self.rel2idx.items()}

        self.tokenizer = BertTokenizer.from_pretrained(load_path)

        if os.path.exists(f"{load_path}/ner_model.pt"):
            self._init_ner_model()
            self.ner_model.load_state_dict(torch.load(f"{load_path}/ner_model.pt", map_location=self.device))
            self.ner_model.to(self.device)

        if os.path.exists(f"{load_path}/rel_model.pt"):
            self._init_rel_model()
            self.rel_model.load_state_dict(torch.load(f"{load_path}/rel_model.pt", map_location=self.device))
            self.rel_model.to(self.device)
