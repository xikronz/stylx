import torch
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


class StylizedResiduals:
    """
    Testing H1 for stylx 
    """
    
    def __init__(self, original_text_path, output_data_path, stripped_text_path):
        """
        Initialize the StylizedResiduals analyzer.
        
        Args:
            original_text_path: Path to CSV file containing stylized sentences
            output_data_path: Path to save output CSV files
            stripped_text_path: Path to text that have been stripped from their "style"
        """
        print("Loading models... This may take a while.")
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
        self.translator = AutoModelForCausalLM.from_pretrained("tencent/Hunyuan-MT-7B", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained("tencent/Hunyuan-MT-7B")
        print("Models loaded successfully!")
        
        self.original_language = "English"
        self.target_languages = ["French", "German", "Dutch"] 
        self.original_text_path = original_text_path
        self.output_data_path = output_data_path
        self.stripped_text_path = stripped_text_path

    def translate_to_target(self, sentence, target_language):
        """
        Translate a sentence to a target language with self.translator and self.tokenizer 
        
        Args:
            sentence: The text to translate
            target_language: Target language name (e.g., "French")
            
        Returns:
            Translated text
        """
        messages = [
            {"role": "user", "content": f"Translate the following segment into {target_language}, without additional explanation.\n\n{sentence}"},
        ]
        
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        input_length = tokenized_chat.shape[1]
        
        outputs = self.translator.generate(
            tokenized_chat.to(self.translator.device), 
            max_new_tokens=2048
        )
        
        #decode only the NEW tokens (skip the input prompt)
        output_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        return output_text.strip() 

    def load_file(self): 
        """
        Load stylized text from CSV file.
        
        Returns:
            DataFrame containing stylized sentences
        """
        stylized_text = pd.read_csv(self.original_text_path)
        print(f"Loaded {len(stylized_text)} sentences from {self.original_text_path}")
        return stylized_text 

    def load_checkpoint(self):
        """
        Load progress from checkpoint file if it exists.
        
        Returns:
            DataFrame with original and unstylized sentences, or None if no checkpoint exists
        """
        checkpoint_path = self.output_data_path.replace('.csv', '_checkpoint.csv')
        
        try:
            checkpoint_df = pd.read_csv(checkpoint_path)
            print(f"Found checkpoint with {len(checkpoint_df)} completed sentences")
            return checkpoint_df
        except FileNotFoundError:
            print("No checkpoint file found. Starting from scratch.")
            return None

    def strip_style(self, stylized_text, resume_from_checkpoint=True): 
        """
        Strip style from sentences by translating through multiple languages.
        Writes progress incrementally to a CSV file as backup.
        
        Args:
            stylized_text: List of stylized sentences
            resume_from_checkpoint: If True, resume from checkpoint if it exists
            
        Returns:
            List of unstylized (round-trip translated) sentences
        """
        unstylized_text = []
        completed_sentences = set()
        
        checkpoint_path = self.output_data_path.replace('.csv', '_checkpoint.csv')
        
        #check for existing checkpoint
        if resume_from_checkpoint:
            checkpoint_df = self.load_checkpoint()
            if checkpoint_df is not None:
                #load already completed sentences
                for _, row in checkpoint_df.iterrows():
                    completed_sentences.add(row['original'])
                    unstylized_text.append(row['unstylized'])
                print(f"Resuming from checkpoint. Skipping {len(completed_sentences)} completed sentences.")
            else:
                #no checkpoint, create new file with header
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    f.write("original,unstylized\n")
        else:
            #start fresh, overwrite any existing checkpoint
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                f.write("original,unstylized\n")
        
        print(f"Progress will be saved to: {checkpoint_path}")
        
        for idx, sentence in enumerate(stylized_text):
            #skip if already completed
            if sentence in completed_sentences:
                print(f"Skipping sentence {idx+1}/{len(stylized_text)} (already completed)")
                continue
            print(f"\n{'='*60}")
            print(f"Processing sentence {idx+1}/{len(stylized_text)}")
            print(f"{'='*60}")
            
            curr = sentence
            
            #translate through each target language sequentially
            for language in self.target_languages:
                print(f"Translating to {language}...")
                curr = self.translate_to_target(curr, language)
            
            print(f"Translating back to {self.original_language}...")
            stripped_sentence = self.translate_to_target(curr, self.original_language)
            
            unstylized_text.append(stripped_sentence)
            with open(checkpoint_path, 'a', encoding='utf-8') as f:
                original_escaped = sentence.replace('"', '""').replace('\n', ' ')
                stripped_escaped = stripped_sentence.replace('"', '""').replace('\n', ' ')
                f.write(f'"{original_escaped}","{stripped_escaped}"\n')
                f.flush() 
            
            print(f"✓ Saved progress: {idx+1}/{len(stylized_text)} sentences completed")

        print(f"\n{'='*60}")
        print(f"All sentences processed! Checkpoint saved to {checkpoint_path}")
        print(f"{'='*60}")
        
        return unstylized_text

    def compute_residuals(self, stylized, unstylized):
        """
        Compute embedding residuals between stylized and unstylized text.
        
        Args:
            stylized: List of stylized sentences
            unstylized: List of unstylized sentences
            
        Returns:
            numpy array of residual vectors
        """
        residuals = []
        
        for i in range(len(stylized)):
            stylized_enc = self.model.encode(stylized[i], convert_to_numpy=True)
            unstylized_enc = self.model.encode(unstylized[i], convert_to_numpy=True)
            
            diff = stylized_enc - unstylized_enc
            residuals.append(diff)

        residuals = np.array(residuals)
        print(f"Computed {len(residuals)} residuals with dimension {residuals.shape[1]}")
        return residuals

    def analyze_magnitudes(self, residuals):
        """
        Analyze the L2 norms of residual vectors.
        
        Args:
            residuals: numpy array of residual vectors
            
        Returns:
            Array of residual norms
        """
        print("\n" + "=" * 60)
        print("RESIDUAL MAGNITUDES")
        print("=" * 60)
        residual_norms = np.linalg.norm(residuals, axis=1)
        
        for i, norm in enumerate(residual_norms):
            print(f"Pair {i+1}: ||residual|| = {norm:.4f}")
        
        return residual_norms

    def compute_cosine_similarities(self, residuals, avg_style_vector):
        """
        Compute cosine similarities between residuals and average style vector.
        
        Args:
            residuals: numpy array of residual vectors
            avg_style_vector: The average style direction vector
            
        Returns:
            Array of cosine similarities
        """
        print("\n" + "=" * 60)
        print("COSINE SIMILARITY WITH AVERAGE STYLE VECTOR")
        print("=" * 60)
        
        cosines = []
        for i, residual in enumerate(residuals):
            cosine_sim = np.dot(residual, avg_style_vector) / (
                np.linalg.norm(residual) * np.linalg.norm(avg_style_vector)
            )
            cosines.append(cosine_sim)
            print(f"Pair {i+1}: cosine similarity = {cosine_sim:.4f}")

        return np.array(cosines)

    def compute_similarity_matrix(self, residuals):
        """
        Compute pairwise cosine similarity matrix between all residuals.
        
        Args:
            residuals: numpy array of residual vectors
            
        Returns:
            Similarity matrix
        """
        print("\n" + "=" * 60)
        print("PAIRWISE COSINE SIMILARITY MATRIX")
        print("=" * 60)
        
        residuals_normalized = normalize(residuals, axis=1)
        similarity_matrix = residuals_normalized @ residuals_normalized.T
        
        mean_off_diag = (similarity_matrix.sum() - len(residuals)) / (len(residuals) * (len(residuals) - 1))
        print(f"Mean off-diagonal similarity: {mean_off_diag:.4f}")
        
        return similarity_matrix

    def compute_dot_products(self, residuals):
        """
        Compute dot product matrix between residuals (includes magnitude info).
        
        Args:
            residuals: numpy array of residual vectors
            
        Returns:
            Dot product matrix
        """
        print("\n" + "=" * 60)
        print("DOT PRODUCTS BETWEEN RESIDUALS")
        print("=" * 60)
        
        dot_product_matrix = residuals @ residuals.T
        return dot_product_matrix

    def compute_average_vector(self, residuals):
        """
        Compute the average style vector across all residuals.
        
        Args:
            residuals: numpy array of residual vectors
            
        Returns:
            Average style vector
        """
        avg_style_vector = residuals.mean(axis=0)
        print(f"\nAverage style vector computed with dimension {len(avg_style_vector)}")
        return avg_style_vector

    def visualize(self, residuals, similarity_matrix, dot_product_matrix, 
                  residual_norms, output_path='style_residuals_analysis.png'):
        """
        Create comprehensive visualization of residual analysis.
        
        Args:
            residuals: numpy array of residual vectors
            similarity_matrix: Pairwise cosine similarity matrix
            dot_product_matrix: Pairwise dot product matrix
            residual_norms: Array of residual magnitudes
            output_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        num_samples = len(residuals)
        
        #plot 1: Cosine Similarity Heatmap
        sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                    center=0.5, vmin=0, vmax=1,
                    xticklabels=[f'Pair {i+1}' for i in range(num_samples)],
                    yticklabels=[f'Pair {i+1}' for i in range(num_samples)],
                    ax=axes[0, 0], cbar_kws={'label': 'Cosine Similarity'})
        axes[0, 0].set_title('Cosine Similarity Between Style Residuals\n(High values = consistent style direction)', 
                              fontsize=12, fontweight='bold')

        #plot 2: Dot Product Heatmap
        sns.heatmap(dot_product_matrix, annot=True, fmt='.1f', cmap='viridis',
                    xticklabels=[f'Pair {i+1}' for i in range(num_samples)],
                    yticklabels=[f'Pair {i+1}' for i in range(num_samples)],
                    ax=axes[0, 1], cbar_kws={'label': 'Dot Product'})
        axes[0, 1].set_title('Dot Products Between Style Residuals\n(Includes magnitude information)', 
                              fontsize=12, fontweight='bold')

        #plot 3: Residual Magnitudes
        colors = plt.cm.viridis(np.linspace(0, 1, num_samples))
        bars = axes[1, 0].bar(range(1, num_samples + 1), residual_norms, color=colors)
        axes[1, 0].set_xlabel('Sentence Pair', fontsize=11)
        axes[1, 0].set_ylabel('L2 Norm', fontsize=11)
        axes[1, 0].set_title('Magnitude of Style Residuals\n(How different are the embeddings?)', 
                              fontsize=12, fontweight='bold')
        axes[1, 0].set_xticks(range(1, num_samples + 1))
        axes[1, 0].grid(axis='y', alpha=0.3)
        for i, (bar, norm) in enumerate(zip(bars, residual_norms)):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + max(residual_norms)*0.01,
                           f'{norm:.2f}', ha='center', va='bottom', fontsize=10)

        #plot 4: PCA visualization of residuals
        if len(residuals) > 1:
            pca = PCA(n_components=min(2, len(residuals)))
            residuals_pca = pca.fit_transform(residuals)
            
            axes[1, 1].scatter(residuals_pca[:, 0], residuals_pca[:, 1], 
                               s=200, c=colors, alpha=0.7, edgecolors='black', linewidth=1.5)
            for i, (x, y) in enumerate(residuals_pca):
                axes[1, 1].annotate(f'Pair {i+1}', (x, y), 
                                   xytext=(10, 10), textcoords='offset points',
                                   fontsize=10, fontweight='bold')
            axes[1, 1].axhline(0, color='gray', linestyle='--', alpha=0.3)
            axes[1, 1].axvline(0, color='gray', linestyle='--', alpha=0.3)
            axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
            if len(pca.explained_variance_ratio_) > 1:
                axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
            axes[1, 1].set_title('PCA of Style Residuals\n(Do they cluster?)', 
                                  fontsize=12, fontweight='bold')
            axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print("\n" + "=" * 60)
        print(f"Visualization saved to '{output_path}'")
        print("=" * 60)
        plt.show()

    def save_to_csv(self, sentences, output_path=None): 
        """
        Save sentences to a CSV file.
        
        Args:
            sentences: List of sentences to save
            output_path: Path to save CSV (defaults to self.output_data_path)
            
        Returns:
            DataFrame that was saved
        """
        if output_path is None:
            output_path = self.output_data_path
            
        dataframe = pd.DataFrame({"sentence": sentences})
        dataframe.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Saved {len(sentences)} sentences to {output_path}")
        return dataframe

    def run_full_analysis(self, stylized_sentences, unstylized_sentences=None):
        """
        Run complete residual analysis pipeline.
        
        Args:
            stylized_sentences: List of stylized sentences
            unstylized_sentences: List of unstylized sentences (optional, will be generated if None)
            
        Returns:
            Dictionary containing all analysis results
        """
        if unstylized_sentences is None:
            print("\nStripping style from sentences...")
            unstylized_sentences = self.strip_style(stylized_sentences)
            self.save_to_csv(unstylized_sentences, 
                           self.output_data_path.replace('.csv', '_unstylized.csv'))
        
        residuals = self.compute_residuals(stylized_sentences, unstylized_sentences)
        
        residual_norms = self.analyze_magnitudes(residuals)
        
        avg_style_vector = self.compute_average_vector(residuals)
        
        cosine_sims = self.compute_cosine_similarities(residuals, avg_style_vector)
        similarity_matrix = self.compute_similarity_matrix(residuals)
        dot_product_matrix = self.compute_dot_products(residuals)
        
        self.visualize(residuals, similarity_matrix, dot_product_matrix, residual_norms)
        
        return {
            'residuals': residuals,
            'residual_norms': residual_norms,
            'avg_style_vector': avg_style_vector,
            'cosine_sims': cosine_sims,
            'similarity_matrix': similarity_matrix,
            'dot_product_matrix': dot_product_matrix
        }


if __name__ == "__main__":
    torch.cuda.empty_cache()

    analyzer = StylizedResiduals(
        original_text_path='kafka_sentences_merged.csv',
        output_data_path='kafka_unstylized_sentences.csv', 
        stripped_text_path='kafka_unstylized_sentences_checkpoint.csv'
    )

    stylized_df = pd.read_csv(analyzer.original_text_path)
    unstyled_df = pd.read_csv(analyzer.stripped_text_path)

    stylized_list = unstyled_df["original"].tolist()
    unstyled_list = unstyled_df["unstylized"].tolist()

    results = analyzer.run_full_analysis(stylized_list, unstyled_list)


    analyzer.save_to_csv(results, 'results.txt')


    
