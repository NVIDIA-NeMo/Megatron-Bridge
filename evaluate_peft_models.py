#!/usr/bin/env python3

"""
Comprehensive PEFT Model Evaluation Script
Supports both NeMo-LM and NeMo2 PEFT models with detailed metrics

Usage:
    python evaluate_peft_models.py --model-path /path/to/checkpoint --approach nemo-lm
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import datasets

# Metric calculation imports
from rouge_score import rouge_scorer
from collections import Counter
import re
import string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PEFTEvaluator:
    """
    Comprehensive PEFT model evaluator that works with both NeMo-LM and NeMo2 models
    """
    
    def __init__(self, model_path: str, approach: str, config: Dict = None):
        self.model_path = Path(model_path)
        self.approach = approach.lower()
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
    def load_model(self):
        """Load PEFT model based on approach"""
        if self.approach == 'nemo-lm':
            return self._load_nemo_lm_model()
        elif self.approach == 'nemo2':
            return self._load_nemo2_model()
        else:
            raise ValueError(f"Unsupported approach: {self.approach}")
            
    def _load_nemo_lm_model(self):
        """Load NeMo-LM PEFT model"""
        # For NeMo-LM, we need to load the base model and apply PEFT adapters
        logger.info("Loading NeMo-LM PEFT model...")
        
        # This would involve loading the checkpoint using NeMo-LM's mechanisms
        # For now, we'll use a HuggingFace approach as a placeholder
        base_model_name = self.config.get('base_model', 'meta-llama/Meta-Llama-3-8B')
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # In a real implementation, you would load the PEFT adapters here
        # from the NeMo-LM checkpoint format
        
        return model, tokenizer
        
    def _load_nemo2_model(self):
        """Load NeMo2 PEFT model"""
        logger.info("Loading NeMo2 PEFT model...")
        
        # Check if this is an HF adapter checkpoint
        hf_adapter_path = self.model_path / "hf_adapter"
        if hf_adapter_path.exists():
            base_model_name = self.config.get('base_model', 'meta-llama/Meta-Llama-3-8B')
            
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            
            # Load PEFT adapter
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(hf_adapter_path))
                logger.info("Successfully loaded PEFT adapter")
            except ImportError:
                logger.warning("PEFT library not available, using base model")
            
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            
        else:
            # Load directly from checkpoint
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            
        return model, tokenizer
        
    def evaluate_on_squad(self, model, tokenizer, num_samples: int = 100) -> Dict:
        """Evaluate model on SQuAD dataset"""
        logger.info(f"Evaluating on SQuAD dataset ({num_samples} samples)")
        
        # Load SQuAD validation dataset
        dataset = datasets.load_dataset("squad", split="validation")
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            
        predictions = []
        references = []
        
        model.eval()
        with torch.no_grad():
            for i, example in enumerate(dataset):
                if i % 20 == 0:
                    logger.info(f"Processing example {i}/{len(dataset)}")
                    
                # Format input
                context = example["context"]
                question = example["question"]
                prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                
                # Tokenize and generate
                inputs = tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    max_length=512, 
                    truncation=True
                ).to(self.device)
                
                # Generate response
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                
                # Extract generated text
                generated_text = tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                predictions.append(generated_text)
                references.append(example["answers"]["text"])
                
        # Calculate metrics
        metrics = self._calculate_qa_metrics(predictions, references)
        
        return {
            'num_samples': len(predictions),
            'predictions': predictions[:5],  # Save first 5 for inspection
            'references': references[:5],
            **metrics
        }
        
    def evaluate_perplexity(self, model, tokenizer, text_samples: List[str]) -> float:
        """Calculate perplexity on text samples"""
        logger.info("Calculating perplexity...")
        
        total_loss = 0.0
        total_tokens = 0
        
        model.eval()
        with torch.no_grad():
            for text in text_samples:
                inputs = tokenizer(
                    text, 
                    return_tensors="pt", 
                    max_length=512, 
                    truncation=True
                ).to(self.device)
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(**inputs, labels=inputs.input_ids)
                    
                total_loss += outputs.loss.item() * inputs.input_ids.shape[1]
                total_tokens += inputs.input_ids.shape[1]
                
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
        
    def benchmark_inference_speed(self, model, tokenizer, num_samples: int = 50) -> Dict:
        """Benchmark inference speed"""
        logger.info(f"Benchmarking inference speed ({num_samples} samples)")
        
        # Prepare test prompts
        test_prompts = [
            "Explain the concept of machine learning in simple terms.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "How does artificial intelligence work?",
            "What is the importance of biodiversity?",
        ] * (num_samples // 5 + 1)
        test_prompts = test_prompts[:num_samples]
        
        times = []
        token_counts = []
        
        model.eval()
        with torch.no_grad():
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                
                start_time = time.time()
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                end_time = time.time()
                
                generation_time = end_time - start_time
                generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
                
                times.append(generation_time)
                token_counts.append(generated_tokens)
                
        # Calculate statistics
        times = np.array(times)
        token_counts = np.array(token_counts)
        tokens_per_second = token_counts / times
        
        return {
            'avg_latency_ms': np.mean(times) * 1000,
            'p95_latency_ms': np.percentile(times, 95) * 1000,
            'avg_tokens_per_second': np.mean(tokens_per_second),
            'total_samples': len(times),
        }
        
    def _calculate_qa_metrics(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """Calculate QA metrics (Exact Match, F1, Rouge-L)"""
        
        exact_matches = 0
        f1_scores = []
        rouge_scores = []
        
        for pred, ref_list in zip(predictions, references):
            # Exact Match
            em_score = max([self._exact_match_score(pred, ref) for ref in ref_list])
            exact_matches += em_score
            
            # F1 Score
            f1_score = max([self._f1_score(pred, ref) for ref in ref_list])
            f1_scores.append(f1_score)
            
            # Rouge-L
            rouge_score = max([
                self.rouge_scorer.score(ref, pred)['rougeL'].fmeasure 
                for ref in ref_list
            ])
            rouge_scores.append(rouge_score)
            
        return {
            'exact_match': (exact_matches / len(predictions)) * 100,
            'f1_score': (sum(f1_scores) / len(f1_scores)) * 100,
            'rouge_l': (sum(rouge_scores) / len(rouge_scores)) * 100,
        }
        
    def _normalize_answer(self, s: str) -> str:
        """Normalize answer for evaluation"""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
            
        def white_space_fix(text):  
            return ' '.join(text.split())
            
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
            
        def lower(text):
            return text.lower()
            
        return white_space_fix(remove_articles(remove_punc(lower(s))))
        
    def _exact_match_score(self, prediction: str, ground_truth: str) -> int:
        """Calculate exact match score"""
        return int(self._normalize_answer(prediction) == self._normalize_answer(ground_truth))
        
    def _f1_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate F1 score"""
        pred_tokens = self._normalize_answer(prediction).split()
        truth_tokens = self._normalize_answer(ground_truth).split()
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)
            
        common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common_tokens.values())
        
        if num_same == 0:
            return 0
            
        precision = num_same / len(pred_tokens)
        recall = num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
        
    def get_model_info(self, model) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_percentage': (trainable_params / total_params) * 100 if total_params > 0 else 0,
            'model_size_mb': (total_params * 4) / (1024 ** 2),  # Assuming fp32
        }
        
    def run_comprehensive_evaluation(self) -> Dict:
        """Run comprehensive evaluation"""
        logger.info("Starting comprehensive PEFT model evaluation")
        
        # Load model
        model, tokenizer = self.load_model()
        
        # Model information
        model_info = self.get_model_info(model)
        logger.info(f"Model loaded: {model_info['total_parameters']:,} total parameters, "
                   f"{model_info['trainable_parameters']:,} trainable ({model_info['trainable_percentage']:.2f}%)")
        
        results = {
            'model_path': str(self.model_path),
            'approach': self.approach,
            'model_info': model_info,
        }
        
        # Evaluate on SQuAD
        try:
            squad_results = self.evaluate_on_squad(model, tokenizer)
            results['squad_evaluation'] = squad_results
            logger.info(f"SQuAD Results - EM: {squad_results['exact_match']:.1f}%, "
                       f"F1: {squad_results['f1_score']:.1f}%, Rouge-L: {squad_results['rouge_l']:.1f}%")
        except Exception as e:
            logger.error(f"SQuAD evaluation failed: {e}")
            results['squad_evaluation'] = {'error': str(e)}
            
        # Benchmark inference speed
        try:
            speed_results = self.benchmark_inference_speed(model, tokenizer)
            results['inference_benchmark'] = speed_results
            logger.info(f"Inference Speed - Avg: {speed_results['avg_tokens_per_second']:.1f} tokens/s, "
                       f"Latency: {speed_results['avg_latency_ms']:.1f}ms")
        except Exception as e:
            logger.error(f"Inference benchmark failed: {e}")
            results['inference_benchmark'] = {'error': str(e)}
            
        # Calculate perplexity on sample texts
        try:
            sample_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Climate change is one of the most pressing issues of our time.",
                "The human brain contains approximately 86 billion neurons.",
                "Renewable energy sources include solar, wind, and hydroelectric power.",
            ]
            perplexity = self.evaluate_perplexity(model, tokenizer, sample_texts)
            results['perplexity'] = perplexity
            logger.info(f"Perplexity: {perplexity:.2f}")
        except Exception as e:
            logger.error(f"Perplexity calculation failed: {e}")
            results['perplexity'] = {'error': str(e)}
            
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PEFT models")
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the PEFT model checkpoint'
    )
    parser.add_argument(
        '--approach',
        choices=['nemo-lm', 'nemo2'],
        required=True,
        help='Which approach was used to train the model'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='meta-llama/Meta-Llama-3-8B',
        help='Base model name for loading'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--squad-samples',
        type=int,
        default=100,
        help='Number of SQuAD samples to evaluate'
    )
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'base_model': args.base_model,
        'squad_samples': args.squad_samples,
    }
    
    # Initialize evaluator
    evaluator = PEFTEvaluator(args.model_path, args.approach, config)
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    # Print summary
    print("\n" + "="*60)
    print("PEFT MODEL EVALUATION SUMMARY")
    print("="*60)
    print(f"Model Path: {results['model_path']}")
    print(f"Approach: {results['approach']}")
    print(f"Total Parameters: {results['model_info']['total_parameters']:,}")
    print(f"Trainable Parameters: {results['model_info']['trainable_parameters']:,} "
          f"({results['model_info']['trainable_percentage']:.2f}%)")
    
    if 'squad_evaluation' in results and 'exact_match' in results['squad_evaluation']:
        squad = results['squad_evaluation']
        print(f"\nSQuAD Results:")
        print(f"  Exact Match: {squad['exact_match']:.1f}%")
        print(f"  F1 Score: {squad['f1_score']:.1f}%")
        print(f"  Rouge-L: {squad['rouge_l']:.1f}%")
        
    if 'inference_benchmark' in results and 'avg_tokens_per_second' in results['inference_benchmark']:
        speed = results['inference_benchmark']
        print(f"\nInference Performance:")
        print(f"  Tokens/second: {speed['avg_tokens_per_second']:.1f}")
        print(f"  Avg Latency: {speed['avg_latency_ms']:.1f}ms")
        print(f"  P95 Latency: {speed['p95_latency_ms']:.1f}ms")
        
    if 'perplexity' in results and isinstance(results['perplexity'], (int, float)):
        print(f"\nPerplexity: {results['perplexity']:.2f}")
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output_file}")
        
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main() 