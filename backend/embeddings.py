"""
Embedding Service for Policy Documents
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Union
import logging
import pickle
import os
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating and managing document embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the embedding model
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized embedding model: {model_name}")
        
        # Model properties
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        show_progress: bool = False
    ) -> Union[np.ndarray, List[List[float]]]:
        """
        Encode text(s) to embeddings
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Embedding(s) as numpy array or list of lists
        """
        try:
            # Handle single text input
            if isinstance(texts, str):
                # Check cache first
                cache_key = self._get_cache_key(texts)
                cached_embedding = self._get_cached_embedding(cache_key)
                if cached_embedding is not None:
                    return cached_embedding
                
                # Generate embedding
                embedding = self.model.encode(
                    texts, 
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True
                )
                
                # Cache the result
                self._cache_embedding(cache_key, embedding)
                
                return embedding
            
            # Handle list of texts
            elif isinstance(texts, list):
                embeddings = []
                uncached_texts = []
                uncached_indices = []
                
                # Check cache for each text
                for i, text in enumerate(texts):
                    cache_key = self._get_cache_key(text)
                    cached_embedding = self._get_cached_embedding(cache_key)
                    
                    if cached_embedding is not None:
                        embeddings.append(cached_embedding)
                    else:
                        embeddings.append(None)  # Placeholder
                        uncached_texts.append(text)
                        uncached_indices.append(i)
                
                # Generate embeddings for uncached texts
                if uncached_texts:
                    new_embeddings = self.model.encode(
                        uncached_texts,
                        batch_size=batch_size,
                        show_progress_bar=show_progress,
                        convert_to_numpy=True
                    )
                    
                    # Update embeddings list and cache
                    for i, embedding in enumerate(new_embeddings):
                        original_index = uncached_indices[i]
                        embeddings[original_index] = embedding
                        
                        # Cache the embedding
                        cache_key = self._get_cache_key(uncached_texts[i])
                        self._cache_embedding(cache_key, embedding)
                
                return embeddings
            
            else:
                raise ValueError("Input must be string or list of strings")
                
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise
    
    def encode_documents(
        self, 
        documents: List[Dict[str, Any]], 
        text_field: str = "text",
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Encode a list of documents with metadata
        
        Args:
            documents: List of document dictionaries
            text_field: Field name containing the text to encode
            batch_size: Batch size for processing
            
        Returns:
            List of documents with added 'embedding' field
        """
        try:
            # Extract texts
            texts = [doc.get(text_field, "") for doc in documents]
            
            # Generate embeddings
            embeddings = self.encode(texts, batch_size=batch_size)
            
            # Add embeddings to documents
            for i, doc in enumerate(documents):
                doc['embedding'] = embeddings[i].tolist() if isinstance(embeddings[i], np.ndarray) else embeddings[i]
            
            logger.info(f"Encoded {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Document encoding failed: {e}")
            raise
    
    def compute_similarity(
        self, 
        embedding1: Union[List[float], np.ndarray], 
        embedding2: Union[List[float], np.ndarray]
    ) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        try:
            # Convert to numpy arrays
            if isinstance(embedding1, list):
                embedding1 = np.array(embedding1)
            if isinstance(embedding2, list):
                embedding2 = np.array(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            raise
    
    def find_similar(
        self, 
        query_embedding: Union[List[float], np.ndarray],
        candidate_embeddings: List[Union[List[float], np.ndarray]],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Find most similar embeddings to query
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar embeddings with scores
        """
        try:
            similarities = []
            
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.compute_similarity(query_embedding, candidate)
                if similarity >= threshold:
                    similarities.append({
                        'index': i,
                        'similarity': similarity,
                        'embedding': candidate
                    })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise
    
    def batch_encode_file(
        self, 
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        batch_size: int = 1000,
        text_field: str = "text"
    ) -> Path:
        """
        Encode texts from a file in batches and save results
        
        Args:
            file_path: Path to input file (JSON or JSONL)
            output_path: Path to save encoded results
            batch_size: Batch size for processing
            text_field: Field name containing text to encode
            
        Returns:
            Path to output file
        """
        try:
            import json
            
            file_path = Path(file_path)
            if output_path is None:
                output_path = file_path.parent / f"{file_path.stem}_encoded.json"
            else:
                output_path = Path(output_path)
            
            # Load documents
            if file_path.suffix == '.jsonl':
                documents = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        documents.append(json.loads(line.strip()))
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    documents = json.load(f)
            
            # Process in batches
            encoded_documents = []
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_encoded = self.encode_documents(batch, text_field)
                encoded_documents.extend(batch_encoded)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{total_batches}")
            
            # Save results
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(encoded_documents, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Encoded {len(documents)} documents, saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(f"{self.model_name}_{text}".encode()).hexdigest()
    
    def _get_cached_embedding(self, cache_key: str) -> Optional[np.ndarray]:
        """Get cached embedding if exists"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
        return None
    
    def _cache_embedding(self, cache_key: str, embedding: np.ndarray):
        """Cache embedding"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def clear_cache(self):
        """Clear embedding cache"""
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Embedding cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if not self.cache_dir.exists():
                return {'cached_embeddings': 0, 'cache_size_mb': 0}
            
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                'cached_embeddings': len(cache_files),
                'cache_size_mb': round(total_size / (1024 * 1024), 2)
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
