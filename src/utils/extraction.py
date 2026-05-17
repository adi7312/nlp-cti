"""BERT-BiLSTM-CRF Automatic Entity and Relation Extraction.

This module implements a deep learning-based approach for named entity recognition
(NER) and relation extraction using BERT embeddings, BiLSTM layers, and CRF decoding.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torchcrf import CRF

class NERDataset(Dataset):
    """Dataset for Named Entity Recognition.

    Attributes:
        texts: List of tokenized text sequences.
        labels: List of corresponding label sequences.
        tokenizer: BERT tokenizer for text processing.
        label2idx: Mapping from label names to indices.
        max_len: Maximum sequence length.
    """

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
        label = self.labels[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            is_split_into_words=True
        )

        # Convert labels to indices
        label_ids = [self.label2idx.get(l, self.label2idx['O']) for l in label]
        label_ids = label_ids[:self.max_len]

        # Pad labels
        label_ids = label_ids + [self.label2idx['O']] * (self.max_len - len(label_ids))
        label_ids = torch.tensor(label_ids, dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_ids
        }


class RelationDataset(Dataset):
    """Dataset for Relation Extraction.

    Attributes:
        texts: List of tokenized text sequences.
        entities: List of entity spans.
        relations: List of relation triples (head, tail, relation_type).
        tokenizer: BERT tokenizer for text processing.
        rel2idx: Mapping from relation names to indices.
        max_len: Maximum sequence length.
    """

    def __init__(self, texts: List[List[str]], entities: List[List[Tuple[int, int, str]]], relations: List[List[Tuple[int, int, str]]],
                tokenizer: BertTokenizer, rel2idx: Dict[str, int], max_len: int = 128) -> None:
        """Initialize relation extraction dataset.

        Args:
            texts: List of tokenized text sequences.
            entities: List of entity spans (start, end, type) for each text.
            relations: List of relation triples (head_idx, tail_idx, rel_type) for each text.
            tokenizer: BERT tokenizer instance.
            rel2idx: Dictionary mapping relation names to indices.
            max_len: Maximum sequence length (default: 128).
        """
        self.texts = texts
        self.entities = entities
        self.relations = relations
        self.tokenizer = tokenizer
        self.rel2idx = rel2idx
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            Dictionary containing input_ids, attention_mask, entity_spans, and relation_matrix.
        """
        text = self.texts[idx]
        entity_spans = self.entities[idx]
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

        # Create entity position matrix (num_entities x seq_len)
        num_entities = len(entity_spans)
        entity_positions = torch.zeros((num_entities, self.max_len))

        for e_idx, (start, end, _) in enumerate(entity_spans):
            if end <= self.max_len:
                entity_positions[e_idx, start:end] = 1

        # Create relation matrix (num_entities x num_entities x num_relations)
        num_relations = len(self.rel2idx)
        relation_matrix = torch.zeros((num_entities, num_entities, num_relations))

        for head_idx, tail_idx, rel_type in relation_triples:
            if head_idx < num_entities and tail_idx < num_entities:
                rel_idx = self.rel2idx.get(rel_type, 0)
                relation_matrix[head_idx, tail_idx, rel_idx] = 1

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'entity_positions': entity_positions,
            'relation_matrix': relation_matrix
        }


class BERTBiLSTMCRF(nn.Module):
    """BERT-BiLSTM-CRF model for Named Entity Recognition.

    Architecture:
        BERT -> BiLSTM -> CRF

    Attributes:
        bert: BERT model for token embeddings.
        bilstm: Bi-directional LSTM layer.
        crf: Conditional Random Field layer.
        dropouts: Dropout layers.
        label2idx: Mapping from label names to indices.
        num_labels: Number of unique labels.
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

    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[List[str]]:
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

    Attributes:
        bert: BERT model for token embeddings.
        bilstm: Bi-directional LSTM layer.
        rel2idx: Mapping from relation names to indices.
        num_relations: Number of unique relations.
    """

    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        rel2idx: Optional[Dict[str, int]] = None,
        max_entities: int = 20
    ) -> None:
        """Initialize BERT-BiLSTM relation extraction model.

        Args:
            bert_model_name: Name of the BERT model to use (default: 'bert-base-uncased').
            hidden_size: Size of BiLSTM hidden state (default: 128).
            num_layers: Number of BiLSTM layers (default: 2).
            dropout: Dropout probability (default: 0.1).
            rel2idx: Dictionary mapping relation names to indices.
            max_entities: Maximum number of entities per document (default: 20).
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
            num_heads=4
        )

        # Relation classification
        self.relation_classifier = nn.Linear(hidden_size * 2 * 2, self.num_relations)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_positions: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the model.

        Args:
            input_ids: Input token IDs (batch_size, seq_len).
            attention_mask: Attention mask (batch_size, seq_len).
            entity_positions: Entity position matrix (batch_size, num_entities, seq_len).
            labels: Optional ground truth relation matrix (batch_size, num_entities, num_entities, num_relations).

        Returns:
            Tuple of (relation_logits, loss). Loss is None if labels not provided.
        """
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

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
        num_entities = entity_positions.size(1)
        entity_reps = []

        for b in range(batch_size):
            entity_pos = entity_positions[b]  # (num_entities, seq_len)
            weighted_embeddings = torch.matmul(
                entity_pos.unsqueeze(1),
                lstm_output[b].unsqueeze(0)
            ).squeeze(1)  # (num_entities, hidden_size * 2)

            # Normalize by entity length
            entity_lengths = entity_pos.sum(dim=1, keepdim=True)
            entity_reps.append(weighted_embeddings / (entity_lengths + 1e-8))

        entity_reps = torch.stack(entity_reps, dim=0)  # (batch_size, num_entities, hidden_size * 2)

        # Entity-aware attention
        entity_reps_attn, _ = self.entity_attention(
            entity_reps, entity_reps, entity_reps
        )
        entity_reps = entity_reps + entity_reps_attn

        # Pair entities for relation classification
        head_reps = entity_reps.unsqueeze(2).expand(-1, -1, num_entities, -1)
        tail_reps = entity_reps.unsqueeze(1).expand(-1, num_entities, -1, -1)

        pair_reps = torch.cat([head_reps, tail_reps], dim=-1)

        # Relation classification
        relation_logits = self.relation_classifier(pair_reps)
        relation_logits = relation_logits.view(batch_size, num_entities, num_entities, self.num_relations)

        if labels is not None:
            # Reshape labels for loss calculation
            loss_fct = nn.CrossEntropyLoss()
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

    Attributes:
        ner_model: BERT-BiLSTM-CRF model for NER.
        rel_model: BERT-BiLSTM model for relation extraction.
        tokenizer: BERT tokenizer.
        device: Device for model training/inference.
    """

    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        ner_hidden_size: int = 128,
        rel_hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: Optional[str] = None
    ) -> None:
        """Initialize entity and relation extractor.

        Args:
            bert_model_name: Name of the BERT model to use (default: 'bert-base-uncased').
            ner_hidden_size: Hidden size for NER BiLSTM (default: 128).
            rel_hidden_size: Hidden size for relation BiLSTM (default: 128).
            num_layers: Number of BiLSTM layers (default: 2).
            dropout: Dropout probability (default: 0.1).
            device: Device for model training/inference. Uses CUDA if available.
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

    def fit_ner(
        self,
        train_texts: List[List[str]],
        train_labels: List[List[str]],
        label2idx: Dict[str, int],
        batch_size: int = 16,
        epochs: int = 10,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100
    ) -> BERTBiLSTMCRF:
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
        self.ner_model = BERTBiLSTMCRF(
            bert_model_name=self.bert_model_name,
            hidden_size=self.ner_hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            label2idx=label2idx
        ).to(self.device)

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

    def predict_entities(
        self,
        texts: List[List[str]]
    ) -> List[List[Tuple[int, int, str]]]:
        """Predict entities in text sequences.

        Args:
            texts: List of tokenized text sequences.

        Returns:
            List of entity spans (start, end, type) for each text.
        """
        if self.ner_model is None:
            raise ValueError("NER model not trained. Call fit_ner() first.")

        self.ner_model.eval()
        dataset = NERDataset(
            texts=texts,
            labels=[['O'] * len(t) for t in texts],  # Dummy labels
            tokenizer=self.tokenizer,
            label2idx=self.label2idx
        )
        dataloader = DataLoader(dataset, batch_size=16)

        all_entities = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                predictions = self.ner_model.decode(input_ids, attention_mask)

                for text, preds in zip(batch['input_ids'], predictions):
                    entities = self._extract_entities_from_predictions(text, preds)
                    all_entities.append(entities)

        return all_entities

    def _extract_entities_from_predictions(
        self,
        input_ids: torch.Tensor,
        predictions: List[str]
    ) -> List[Tuple[int, int, str]]:
        """Extract entity spans from predicted labels.

        Args:
            input_ids: Input token IDs.
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
                    current_entity = (i, i + 1, label[2:])
            else:
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = None

        if current_entity is not None:
            entities.append(current_entity)

        return entities

    def fit_relation(
        self,
        train_texts: List[List[str]],
        train_entities: List[List[Tuple[int, int, str]]],
        train_relations: List[List[Tuple[int, int, str]]],
        rel2idx: Dict[str, int],
        batch_size: int = 8,
        epochs: int = 15,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100
    ) -> BERTBiLSTMRelationModel:
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
            rel2idx=rel2idx
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.rel_model = BERTBiLSTMRelationModel(
            bert_model_name=self.bert_model_name,
            hidden_size=self.rel_hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            rel2idx=rel2idx
        ).to(self.device)

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
                relation_matrix = batch['relation_matrix'].to(self.device)

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

    def predict_relations(
        self,
        texts: List[List[str]],
        entities: List[List[Tuple[int, int, str]]]
    ) -> List[List[Tuple[int, int, str]]]:
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
            rel2idx=self.rel2idx
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

    def _extract_relations_from_logits(
        self,
        relation_logits: torch.Tensor
    ) -> List[Tuple[int, int, str]]:
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

    def extract(
        self,
        text: str
    ) -> Dict[str, Any]:
        """Extract entities and relations from raw text.

        Args:
            text: Raw input text.

        Returns:
            Dictionary containing extracted entities and relations.
        """
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)

        if self.ner_model:
            entities = self.predict_entities([tokens])[0]
        else:
            entities = []

        if self.rel_model and entities:
            relations = self.predict_relations([tokens], [entities])[0]
        else:
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

    def save(self, save_path: str) -> None:
        """Save models to disk.

        Args:
            save_path: Directory to save models.
        """
        import os
        os.makedirs(save_path, exist_ok=True)

        if self.ner_model:
            torch.save(self.ner_model.state_dict(), f"{save_path}/ner_model.pt")
        if self.rel_model:
            torch.save(self.rel_model.state_dict(), f"{save_path}/rel_model.pt")

        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)

    def load(self, load_path: str) -> None:
        """Load models from disk.

        Args:
            load_path: Directory containing saved models.
        """
        from transformers import BertTokenizer

        self.tokenizer = BertTokenizer.from_pretrained(load_path)

        if self.ner_model:
            self.ner_model.load_state_dict(torch.load(f"{load_path}/ner_model.pt"))
            self.ner_model.to(self.device)

        if self.rel_model:
            self.rel_model.load_state_dict(torch.load(f"{load_path}/rel_model.pt"))
            self.rel_model.to(self.device)
