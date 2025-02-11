import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from datetime import datetime
import networkx as nx
import pickle 
import os 
from typing import List
import gc
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path 
from tqdm import tqdm
import scipy
import faiss
import faiss.contrib.torch_utils  # Enable GPU support for FAISS
import warnings
warnings.filterwarnings('ignore')


class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

class LegalCaseAnalyzer:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            print("Initializing LegalCaseAnalyzer...")
            self.initialized = True
            self.model_dir = Path("saved_models")
            self.model_dir.mkdir(exist_ok=True)
            
            # Force CUDA detection
            if torch.cuda.is_available():
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                self.device = torch.device('cuda:0')
                torch.cuda.empty_cache()
                # Enable tensor cores for faster computation
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
            else:
                print("CUDA not available, using CPU")
                self.device = torch.device('cpu')
            
            if self._load_cached_data():
                print("Loaded pre-computed data successfully!")
                return
                
            print("No cached data found, computing from scratch...")
            self.nlp = spacy.load("en_core_web_sm")
            self.embedder = SentenceTransformer('all-mpnet-base-v2')
            self.embedder.to(self.device)
            
            self.df = self._load_and_preprocess_data()
            self.semantic_index = self._build_semantic_index()
            self.citation_graph = self._build_citation_graph()
            self.keyword_index = self._build_keyword_index()

            if not hasattr(self, 'citation_scores'):
                print("Calculating citation scores...")
                self.citation_scores = self._calculate_citation_scores(self.citation_graph)
            
            self._cache_data()

    def initialize_models(self):
        """Initialize or reinitialize the models"""
        print("Initializing models...")
        
        # Initialize spaCy
        if not hasattr(self, 'nlp'):
            self.nlp = spacy.load("en_core_web_sm")
            
        # Initialize sentence transformer
        if not hasattr(self, 'embedder'):
            self.embedder = SentenceTransformer('all-mpnet-base-v2')
            if torch.cuda.is_available():
                print(f"Moving embedder to GPU: {torch.cuda.get_device_name(0)}")
                self.embedder.to(self.device)
        
        # Initialize other components if needed
        if not all(hasattr(self, attr) for attr in ['df', 'semantic_index', 'citation_graph', 'keyword_index']):
            if not hasattr(self, 'df'):
                self.df = self._load_and_preprocess_data()
            if not hasattr(self, 'semantic_index'):
                self.semantic_index = self._build_semantic_index()
            if not hasattr(self, 'citation_graph'):
                self.citation_graph = self._build_citation_graph()
            if not hasattr(self, 'keyword_index'):
                self.keyword_index = self._build_keyword_index()
        
        print("Models initialized successfully!")

    def _cache_data(self):
        """Save computed data to disk"""
        print("Caching computed data...")
        self.model_dir.mkdir(exist_ok=True)
        
        with open(self.model_dir / 'df.pkl', 'wb') as f:
            pickle.dump(self.df, f)
        
        faiss.write_index(self.semantic_index, str(self.model_dir / 'semantic_index.faiss'))
        
        with open(self.model_dir / 'citation_graph.pkl', 'wb') as f:
            pickle.dump(self.citation_graph, f)
            
        scipy.sparse.save_npz(str(self.model_dir / 'keyword_index.npz'), self.keyword_index)
        
        print("Data cached successfully!")


    def _load_cached_data(self):
        """Try to load pre-computed data from disk"""
        try:
            print("Attempting to load cached data...")
            with open(self.model_dir / 'df.pkl', 'rb') as f:
                self.df = pickle.load(f)
            
            self.semantic_index = faiss.read_index(str(self.model_dir / 'semantic_index.faiss'))
            
            with open(self.model_dir / 'citation_graph.pkl', 'rb') as f:
                self.citation_graph = pickle.load(f)
                
            self.keyword_index = scipy.sparse.load_npz(str(self.model_dir / 'keyword_index.npz'))
            
            return True
            
        except Exception as e:
            print(f"Could not load cached data: {e}")
            return False


    def _load_and_preprocess_data(self):
        print("Loading and preprocessing data...")
        df = pd.read_csv("../datasets/all_tech_cases_5year.csv")
        
        df['combined_text'] = (
            df['case_name'].fillna('') + " " +
            df['text_excerpt'].fillna('') + " " +
            df['tech_keywords_found'].fillna('') + " " +
            df['suit_nature'].fillna('')
        )
        
        df['date_filed'] = pd.to_datetime(df['date_filed'])
        
        try:
            df['tech_relevance_category'] = pd.qcut(
                df['tech_relevance_score'].fillna(0), 
                q=5, 
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
        except ValueError:
            print("Warning: Could not create tech relevance categories. Using default values.")
            df['tech_relevance_category'] = 'Unknown'
        
        return df

    def _compute_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Optimized embedding computation for RTX 3070"""
        # Configure for RTX 3070 (8GB VRAM)
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            # Reserve 1GB for model and other operations
            available_mem = (gpu_mem - (1 * 1024**3)) / 2
            # Estimate memory per sample (approximately 768 * 4 bytes for float32)
            mem_per_sample = 768 * 4  # embedding dimension * bytes per float
            max_batch_size = int(available_mem / mem_per_sample)
            batch_size = min(256, max_batch_size)  # Cap at 256 for stability
            print(f"Using batch size: {batch_size} for GPU processing")
        else:
            batch_size = 32
            print("Using CPU processing with batch size: 32")

        embeddings_list = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        # Pre-allocate CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.memory.empty_cache()
        
        print(f"Processing {len(texts)} texts in {total_batches} batches...")
        
        try:
            for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Computing embeddings"):
                batch = texts[i:i + batch_size]
                
                with torch.no_grad(), torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                    # Process in half precision on GPU
                    batch_embeddings = self.embedder.encode(
                        batch,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        normalize_embeddings=True  # Ensures normalized embeddings
                    )
                    
                    # Move to CPU and convert back to float32
                    embeddings_list.append(batch_embeddings.cpu().float().numpy())
                
                # Memory management
                if torch.cuda.is_available():
                    if i % (batch_size * 10) == 0:  # Print memory usage every 10 batches
                        allocated = torch.cuda.memory_allocated() / 1024**2
                        reserved = torch.cuda.memory_reserved() / 1024**2
                        print(f"\nGPU Memory - Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")
                    torch.cuda.empty_cache()
                    
        except RuntimeError as e:
            if "out of memory" in str(e) and torch.cuda.is_available():
                print("GPU out of memory. Reducing batch size and retrying...")
                torch.cuda.empty_cache()
                # Reduce batch size and try again
                return self._compute_embeddings_batch(texts, batch_size=batch_size // 2)
            raise e

        # Combine all embeddings
        all_embeddings = np.vstack(embeddings_list)
        
        # Clean up
        del embeddings_list
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return all_embeddings

    def _build_semantic_index(self):
        """Rebuild semantic index with proper document handling"""
        print("Building semantic index...")
        
        # First, deduplicate cases
        self.df = self.df.drop_duplicates(subset=['case_name', 'date_filed'], keep='first')
        self.df = self.df.reset_index(drop=True)
        
        texts = self.df['combined_text'].tolist()
        print(f"Processing {len(texts)} unique documents...")
        
        # Compute embeddings
        print("Computing embeddings...")
        embeddings = self._compute_embeddings_batch(texts)
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        # Build index with proper configuration
        print("Building FAISS index...")
        embedding_dim = embeddings.shape[1]
        
        if torch.cuda.is_available():
            try:
                # Configure GPU resources
                res = faiss.StandardGpuResources()
                
                # Build a flat index first
                quantizer = faiss.IndexFlatL2(embedding_dim)
                
                # Create an IVF index
                nlist = min(4096, int(np.sqrt(len(texts))))  # number of clusters
                index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)
                
                # Convert to GPU index
                gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
                
                # Train the index
                print(f"Training index with {len(embeddings)} vectors...")
                gpu_index.train(embeddings.astype(np.float32))
                
                # Add vectors
                print("Adding vectors to index...")
                gpu_index.add(embeddings.astype(np.float32))
                
                # Convert back to CPU for storage
                index = faiss.index_gpu_to_cpu(gpu_index)
                print(f"Built index with {index.ntotal} vectors")
                
                # Set search parameters
                index.nprobe = min(64, nlist)  # number of clusters to visit during search
                
            except Exception as e:
                print(f"GPU index creation failed: {e}. Falling back to CPU index.")
                # Create a more sophisticated CPU index
                index = faiss.IndexPQ(embedding_dim, 8, 8)  # 8 sub-quantizers with 8 bits each
                index.train(embeddings.astype(np.float32))
                index.add(embeddings.astype(np.float32))
        else:
            print("Using CPU index...")
            # Create a more sophisticated CPU index
            index = faiss.IndexPQ(embedding_dim, 8, 8)  # 8 sub-quantizers with 8 bits each
            index.train(embeddings.astype(np.float32))
            index.add(embeddings.astype(np.float32))
        
        # Verify index
        print("Verifying index...")
        sample_query = embeddings[0].reshape(1, -1).astype(np.float32)
        D, I = index.search(sample_query, 5)
        print(f"Sample search results - distances: {D[0]}, indices: {I[0]}")
        
        # Clean up
        del embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return index




    def _build_citation_graph(self):
        """Build citation graph with optimized batch processing"""
        print("Building citation graph...")
        G = nx.DiGraph()
        
        # Pre-sort cases by date for faster citation processing
        print("Preprocessing case dates...")
        date_sorted_df = self.df.sort_values('date_filed')
        date_sorted_df['date_filed'] = pd.to_datetime(date_sorted_df['date_filed'])
        
        # Create date index for faster lookups
        print("Creating date index...")
        date_index = date_sorted_df.reset_index().set_index('date_filed')
        
        # Add all nodes first (faster than batching)
        print("Adding nodes...")
        node_data = {
            idx: {
                'case_name': row['case_name'],
                'date': row['date_filed']
            }
            for idx, row in self.df.iterrows()
        }
        G.add_nodes_from(node_data.items())
        
        # Process citations if available
        if 'citation_count' in self.df.columns:
            print("Processing citations...")
            batch_size = 5000  # Increased batch size
            total_batches = len(self.df) // batch_size + (1 if len(self.df) % batch_size else 0)
            
            # Pre-calculate potential citations for each date
            potential_citations_cache = {}
            
            for start_idx in tqdm(range(0, len(self.df), batch_size), total=total_batches, desc="Processing citation batches"):
                end_idx = min(start_idx + batch_size, len(self.df))
                batch = self.df.iloc[start_idx:end_idx]
                
                # Process each case in the batch
                for idx, row in batch.iterrows():
                    if pd.notna(row['citation_count']) and row['citation_count'] > 0:
                        citation_count = int(row['citation_count'])
                        if citation_count > 0:
                            case_date = pd.to_datetime(row['date_filed'])
                            
                            # Use cached potential citations if available
                            if case_date not in potential_citations_cache:
                                potential_citations = date_sorted_df[
                                    date_sorted_df['date_filed'] < case_date
                                ].index.tolist()
                                potential_citations_cache[case_date] = potential_citations
                            else:
                                potential_citations = potential_citations_cache[case_date]
                            
                            # Add edges for citations
                            if potential_citations:
                                # Take only as many citations as needed
                                actual_citations = potential_citations[:min(citation_count, len(potential_citations))]
                                G.add_edges_from((idx, cite_idx) for cite_idx in actual_citations)
                
                # Clear cache periodically to manage memory
                if len(potential_citations_cache) > 1000:
                    potential_citations_cache.clear()
        
        print(f"Citation graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def _calculate_citation_scores(self, G):
        """Pre-calculate citation importance scores"""
        print("Calculating citation importance scores...")
        try:
            # Calculate PageRank with parallel processing
            pagerank_scores = nx.pagerank(
                G,
                alpha=0.85,  # Damping factor
                max_iter=100,
                tol=1e-06,
                weight=None
            )
            return pagerank_scores
        except Exception as e:
            print(f"Error calculating PageRank: {e}")
            return {node: 1.0/G.number_of_nodes() for node in G.nodes()}


    def _build_keyword_index(self):
        print("Building keyword index...")
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Process in larger batches due to available RAM
        batch_size = 5000
        all_vectors = None
        
        for i in tqdm(range(0, len(self.df), batch_size), desc="Processing keywords"):
            batch = self.df['combined_text'].iloc[i:i + batch_size]
            if all_vectors is None:
                all_vectors = vectorizer.fit_transform(batch)
            else:
                all_vectors = scipy.sparse.vstack([
                    all_vectors, 
                    vectorizer.transform(batch)
                ])
        
        return all_vectors

    def _extract_legal_entities(self, text):
        doc = self.nlp(text)
        entities = {
            'PERSON': [],
            'ORG': [],
            'LAW': [],
            'DATE': []
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        
        return entities

    def search(self, query, k=5):
        """Debug version of search with additional logging and checks"""
        print(f"\nStarting search for query: {query}")
        
        # Compute query embedding
        query_embedding = self.embedder.encode([query]).astype(np.float32)
        
        # Increase initial k and add debug logging
        search_k = k * 20  # Significantly increase the pool
        print(f"Searching for {search_k} initial candidates...")
        
        # Semantic search with debug info
        distances, indices = self.semantic_index.search(query_embedding, search_k)
        print(f"Found {len(indices[0])} initial matches")
        
        # Debug print first few raw results
        print("\nFirst 10 raw matches:")
        for i, (idx, dist) in enumerate(zip(indices[0][:10], distances[0][:10])):
            case = self.df.iloc[idx]
            print(f"{i+1}. {case['case_name']} (Distance: {dist:.4f})")
        
        # Store seen cases with more information
        seen_cases = {}  # Use dict to track why cases were excluded
        results = []
        
        print("\nProcessing candidates...")
        for idx, distance in zip(indices[0], distances[0]):
            case = self.df.iloc[idx]
            case_name = case["case_name"]
            
            # Track duplicate detection
            normalized_name = ' '.join(case_name.lower().split())
            if normalized_name in seen_cases:
                print(f"Skipping duplicate: {case_name}")
                continue
                
            # Verify the case has required fields
            required_fields = ['text_excerpt', 'tech_keywords_found', 'tech_relevance_score']
            if not all(pd.notna(case.get(field, None)) for field in required_fields):
                print(f"Skipping case with missing fields: {case_name}")
                continue
            
            try:
                # Citation importance with fallback
                citation_importance = nx.pagerank(self.citation_graph).get(idx, 0)
                
                # Entity processing with debug info
                case_entities = self._extract_legal_entities(case['text_excerpt'])
                query_entities = self._extract_legal_entities(query)
                
                entity_overlap = sum(
                    len(set(query_entities[ent_type]) & set(case_entities[ent_type]))
                    for ent_type in query_entities
                ) / max(1, sum(len(entities) for entities in query_entities.values()))
                
                # Keyword matching with debug info
                query_words = set(query.lower().split())
                case_text = ' '.join([
                    str(case['case_name']).lower(),
                    str(case['text_excerpt']).lower(),
                    str(case['tech_keywords_found']).lower()
                ])
                case_words = set(case_text.split())
                matching_keywords = query_words & case_words
                keyword_overlap = len(matching_keywords) / len(query_words)
                print(f"\nMatching keywords for {case_name}: {matching_keywords}")
                
                # Technology relevance scoring
                tech_score = float(case['tech_relevance_score'])
                
                # Compute date-based relevance
                try:
                    case_date = pd.to_datetime(case["date_filed"])
                    max_date = pd.to_datetime('2025-01-01')
                    min_date = pd.to_datetime('2000-01-01')
                    date_range = (max_date - min_date).total_seconds()
                    time_relevance = (case_date - min_date).total_seconds() / date_range
                except:
                    time_relevance = 0.5
                
                # Calculate final score
                final_score = (
                    0.3 * (1 - distance) +          # Semantic similarity
                    0.2 * keyword_overlap +         # Direct keyword matching
                    0.2 * tech_score +              # Technology relevance
                    0.15 * citation_importance +    # Citation importance
                    0.1 * entity_overlap +          # Entity overlap
                    0.05 * time_relevance          # Time relevance
                )
                
                result = {
                    "rank": len(results) + 1,
                    "case_name": case_name,
                    "date_filed": case["date_filed"].strftime("%Y-%m-%d"),
                    "court": case["court"],
                    "tech_relevance": case["tech_relevance_category"],
                    "excerpt": case["text_excerpt"][:300],
                    "tech_keywords": case["tech_keywords_found"],
                    "similarity_score": float(final_score),
                    "entities_found": case_entities,
                    "score_breakdown": {
                        "semantic_similarity": float(1 - distance),
                        "citation_importance": float(citation_importance),
                        "entity_overlap": float(entity_overlap),
                        "keyword_overlap": float(keyword_overlap),
                        "tech_relevance": float(tech_score),
                        "time_relevance": float(time_relevance),
                        "matching_keywords": list(matching_keywords)
                    }
                }
                
                results.append(result)
                seen_cases[normalized_name] = True
                print(f"Added case: {case_name} (Score: {final_score:.4f})")
                
            except Exception as e:
                print(f"Error processing case {case_name}: {str(e)}")
                continue
            
            if len(results) >= k:
                break
        
        print(f"\nFound {len(results)} final results")
        
        # Sort by final score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Update ranks after sorting
        for i, result in enumerate(results, 1):
            result["rank"] = i
        
        return results