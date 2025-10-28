import torch
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.utils import resample
from scipy import stats
import re

import os 
os.chdir('..')

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
        print("Loading embedding model... This may take a while.")
        self.preferred_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1", device=self.preferred_device)
        #lazy-load translator to avoid occupying GPU unless needed
        self.translator = None
        self.tokenizer = None
        print("Embedding model loaded successfully!")
        
        self.original_language = "English"
        self.target_languages = ["French", "German", "Dutch"] 
        self.original_text_path = original_text_path
        self.output_data_path = output_data_path
        self.stripped_text_path = stripped_text_path
        
        #analysis/visualization scalability settings
        self.max_heatmap_sample_size = 200  # cap for pairwise heatmaps
        self.max_scatter_points = 5000      # cap before switching to hexbin
        self.topk_neighbors = 10            # for local similarity stats
        self.random_state = 42
        self.max_occlusion_pairs = 25
        self.quiver_max_arrows = 800

    def _ensure_translator_loaded(self):
        if self.translator is None or self.tokenizer is None:
            print("Loading translation model on demand...")
            #prefer CPU for translation to keep GPU memory for embeddings
            self.translator = AutoModelForCausalLM.from_pretrained("tencent/Hunyuan-MT-7B", device_map="cpu")
            self.tokenizer = AutoTokenizer.from_pretrained("tencent/Hunyuan-MT-7B")
            print("Translation model loaded.")

    def translate_to_target(self, sentence, target_language):
        """
        Translate a sentence to a target language with self.translator and self.tokenizer 
        
        Args:
            sentence: The text to translate
            target_language: Target language name (e.g., "French")
            
        Returns:
            Translated text
        """
        self._ensure_translator_loaded()
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

    def _encode_with_retry(self, texts, batch_size=64, show_progress_bar=True):
        """
        Encode texts with automatic batch size reduction and CPU fallback on CUDA OOM.
        """
        bs = max(1, batch_size)
        device_now = self.model.device if hasattr(self.model, 'device') else self.preferred_device
        while True:
            try:
                return self.model.encode(
                    texts,
                    batch_size=bs,
                    convert_to_numpy=True,
                    show_progress_bar=show_progress_bar,
                )
            except RuntimeError as e:
                oom = isinstance(e, torch.cuda.OutOfMemoryError) or 'CUDA out of memory' in str(e)
                if oom:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    if bs > 1:
                        bs = max(1, bs // 2)
                        print(f"CUDA OOM during encode. Reducing batch size to {bs} and retrying...")
                        continue
                    #move to CPU and retry
                    if self.preferred_device == 'cuda':
                        print("CUDA OOM at batch_size=1. Falling back to CPU for encoding...")
                        try:
                            self.model.to('cpu')
                        except Exception:
                            pass
                        self.preferred_device = 'cpu'
                        continue
                raise

    def compute_residuals(self, stylized, unstylized, batch_size=64):
        """
        Compute embedding residuals between stylized and unstylized text.
        
        Args:
            stylized: List of stylized sentences
            unstylized: List of unstylized sentences
            
        Returns:
            numpy array of residual vectors
        """
        #batch encode for scalability
        stylized_enc = self._encode_with_retry(stylized, batch_size=batch_size, show_progress_bar=True)
        unstylized_enc = self._encode_with_retry(unstylized, batch_size=batch_size, show_progress_bar=True)
        residuals = stylized_enc - unstylized_enc
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
        
        if len(residual_norms) <= 50:
            for i, norm in enumerate(residual_norms):
                print(f"Pair {i+1}: ||residual|| = {norm:.4f}")
        else:
            q = np.quantile(residual_norms, [0.0, 0.25, 0.5, 0.75, 1.0])
            print(
                f"Count={len(residual_norms)} | mean={residual_norms.mean():.4f} "+
                f"std={residual_norms.std():.4f} | min={q[0]:.4f} q25={q[1]:.4f} "+
                f"median={q[2]:.4f} q75={q[3]:.4f} max={q[4]:.4f}"
            )
        
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
        
        #vectorized cosine similarity
        denom = (np.linalg.norm(residuals, axis=1) * (np.linalg.norm(avg_style_vector) + 1e-12))
        cosines = (residuals @ avg_style_vector) / np.clip(denom, 1e-12, None)

        if len(cosines) <= 50:
            for i, cosine_sim in enumerate(cosines):
                print(f"Pair {i+1}: cosine similarity = {cosine_sim:.4f}")
        else:
            q = np.quantile(cosines, [0.0, 0.25, 0.5, 0.75, 1.0])
            print(
                f"Count={len(cosines)} | mean={cosines.mean():.4f} std={cosines.std():.4f} "
                f"min={q[0]:.4f} q25={q[1]:.4f} median={q[2]:.4f} q75={q[3]:.4f} max={q[4]:.4f}"
            )

        return cosines

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
        
        n = len(residuals)
        if n > self.max_heatmap_sample_size:
            print(
                f"Dataset too large for full pairwise matrix (n={n}). "
                f"Skipping full computation; will sample in visualization."
            )
            return None
        residuals_normalized = normalize(residuals, axis=1)
        similarity_matrix = residuals_normalized @ residuals_normalized.T
        
        mean_off_diag = (similarity_matrix.sum() - n) / (n * (n - 1)) if n > 1 else 1.0
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
        
        n = len(residuals)
        if n > self.max_heatmap_sample_size:
            print(
                f"Dataset too large for full dot-product matrix (n={n}). "
                f"Skipping full computation; will sample in visualization."
            )
            return None
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

    def _stratified_sample_indices(self, values, sample_size, bins=10, random_state=42):
        """
        Stratified sample indices based on the distribution of `values`.
        """
        rng = np.random.default_rng(random_state)
        if sample_size >= len(values):
            return np.arange(len(values))
        quantiles = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
        indices = np.arange(len(values))
        sampled = []
        per_bin = max(1, sample_size // bins)
        for b in range(bins):
            mask = (values >= quantiles[b]) & (values <= quantiles[b + 1] + 1e-12)
            bin_indices = indices[mask]
            if len(bin_indices) == 0:
                continue
            take = min(per_bin, len(bin_indices))
            chosen = rng.choice(bin_indices, size=take, replace=False)
            sampled.append(chosen)
        sampled = np.concatenate(sampled) if len(sampled) else rng.choice(indices, size=sample_size, replace=False)
        #if we sampled slightly less due to empty bins, top up randomly
        if len(sampled) < sample_size:
            remaining = np.setdiff1d(indices, sampled)
            top_up = rng.choice(remaining, size=sample_size - len(sampled), replace=False)
            sampled = np.concatenate([sampled, top_up])
        return sampled

    def compute_topk_neighbor_similarities(self, residuals, k=None):
        """
        Compute top-k cosine neighbor similarities for each residual without full N^2.
        Returns similarities (n, k).
        """
        if k is None:
            k = self.topk_neighbors
        k = min(k, max(1, len(residuals) - 1))
        residuals_norm = normalize(residuals, axis=1)
        nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
        nn.fit(residuals_norm)
        distances, indices = nn.kneighbors(residuals_norm, return_distance=True)
        #skip self neighbor at position 0
        distances = distances[:, 1:]
        sims = 1.0 - distances
        return sims

    def encode_pairs(self, stylized, unstylized, batch_size=64, show_progress_bar=True):
        """
        Encode stylized and unstylized sentences and return embeddings and residuals.
        Returns (stylized_enc, unstylized_enc, residuals).
        """
        stylized_enc = self._encode_with_retry(stylized, batch_size=batch_size, show_progress_bar=show_progress_bar)
        unstylized_enc = self._encode_with_retry(unstylized, batch_size=batch_size, show_progress_bar=show_progress_bar)
        residuals = stylized_enc - unstylized_enc
        print(f"Encoded pairs: {len(residuals)} residuals with dim {residuals.shape[1]}")
        return stylized_enc, unstylized_enc, residuals

    def linear_probe_style_direction(self, stylized_enc, unstylized_enc, residuals, avg_style_vector=None):
        """
        Train a linear probe to separate stylized vs unstylized embeddings.
        Returns dict with supervised direction, cosine to avg vector, and energy stats.
        """
        X = np.vstack([stylized_enc, unstylized_enc])
        y = np.array([1] * len(stylized_enc) + [0] * len(unstylized_enc))
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        w = clf.coef_.ravel()
        w_norm = np.linalg.norm(w) + 1e-12
        w = w / w_norm
        if avg_style_vector is None:
            avg_style_vector = residuals.mean(axis=0)
        v = avg_style_vector / (np.linalg.norm(avg_style_vector) + 1e-12)
        cos_wa = float(np.clip(w @ v, -1.0, 1.0))
        on_w = residuals @ w
        on_v = residuals @ v
        energy_w = float(np.mean(on_w ** 2))
        energy_v = float(np.mean(on_v ** 2))
        return {
            'w': w,
            'cos_w_vs_avg': cos_wa,
            'energy_on_w': energy_w,
            'energy_on_avg': energy_v,
        }

    def svd_residual_analysis(self, residuals):
        """
        Perform SVD on mean-centered residuals. Return singular values, explained variance,
        cumulative explained, and participation ratio.
        """
        R = residuals - residuals.mean(axis=0, keepdims=True)
        U_svd, s, Vt = np.linalg.svd(R, full_matrices=False)
        power = s ** 2
        total = float(np.sum(power)) + 1e-12
        explained = power / total
        cumulative = np.cumsum(explained)
        participation_ratio = float((np.sum(explained) ** 2) / (np.sum(explained ** 2) + 1e-12))
        return {
            'singular_values': s,
            'explained': explained,
            'cumulative': cumulative,
            'participation_ratio': participation_ratio,
        }

    def angle_diagnostics(self, residuals, directions_dict):
        """
        Compute angle distributions (degrees) between residuals and provided unit directions.
        directions_dict: {name: vector}
        Returns {name: angles_deg_array}
        """
        res_unit = residuals / (np.linalg.norm(residuals, axis=1, keepdims=True) + 1e-12)
        out = {}
        for name, vec in directions_dict.items():
            v = vec / (np.linalg.norm(vec) + 1e-12)
            cos = np.clip(res_unit @ v, -1.0, 1.0)
            angles_deg = np.degrees(np.arccos(cos))
            out[name] = angles_deg
        return out

    def plot_quiver_arrows(self, stylized_enc, unstylized_enc, output_path='style_flow_quiver.png'):
        """
        2D PCA projection with arrows from unstylized to stylized embeddings.
        """
        if len(unstylized_enc) == 0:
            return
        X = np.vstack([unstylized_enc, stylized_enc])
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X)
        n = len(unstylized_enc)
        U2 = X2[:n]
        S2 = X2[n:]
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n, size=min(self.quiver_max_arrows, n), replace=False)
        plt.figure(figsize=(8, 6))
        plt.quiver(U2[idx, 0], U2[idx, 1], (S2 - U2)[idx, 0], (S2 - U2)[idx, 1],
                   angles='xy', scale_units='xy', scale=1, width=0.002, alpha=0.35)
        plt.scatter(U2[idx, 0], U2[idx, 1], s=8, c='gray', alpha=0.5)
        plt.title('Quiver: Unstylized → Stylized (PCA-2D)')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved quiver plot to '{output_path}'")
        plt.close()

    def _has_old_english(self, text, lexicon=None):
        if lexicon is None:
            lexicon = {"thou", "thee", "thy", "thine", "hath", "dost", "shalt", "wherefore", "nay"}
        for w in lexicon:
            if re.search(rf"\b{re.escape(w)}\b", text, flags=re.IGNORECASE):
                return True
        return False

    def lexicon_conditioned_effects(self, texts_unstylized, residuals, direction, n_permutations=2000):
        """
        Compare projection along a style direction for sentences with vs without lexicon words.
        Returns dict with means, diff, t-test p, and permutation p.
        """
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        proj = residuals @ direction
        mask = np.array([self._has_old_english(t) for t in texts_unstylized])
        a = proj[mask]
        b = proj[~mask]
        if len(a) == 0 or len(b) == 0:
            return {'count_with': int(len(a)), 'count_without': int(len(b)), 'diff_mean': None, 't_p': None, 'perm_p': None}
        diff = float(a.mean() - b.mean())
        t_p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        # permutation test on difference in means
        rng = np.random.default_rng(self.random_state)
        combined = np.concatenate([a, b])
        n_a = len(a)
        more_extreme = 0
        for _ in range(n_permutations):
            rng.shuffle(combined)
            a_s = combined[:n_a]
            b_s = combined[n_a:]
            if abs(a_s.mean() - b_s.mean()) >= abs(diff):
                more_extreme += 1
        perm_p = (more_extreme + 1) / (n_permutations + 1)
        return {'count_with': int(len(a)), 'count_without': int(len(b)), 'diff_mean': diff, 't_p': t_p, 'perm_p': float(perm_p)}

    def token_occlusion_attribution(self, stylized_texts, unstylized_texts, direction, num_pairs=None, output_path='token_attribution.png'):
        """
        For a sample of pairs, remove each token in the unstylized text and measure drop
        in cosine alignment of residual with style direction. Aggregate token contributions.
        """
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        rng = np.random.default_rng(self.random_state)
        n = len(unstylized_texts)
        if num_pairs is None:
            num_pairs = min(self.max_occlusion_pairs, n)
        idx = rng.choice(n, size=num_pairs, replace=False)
        token_contribs = {}
        for i in idx:
            t_u = unstylized_texts[i]
            t_s = stylized_texts[i]
            base_u = self._encode_with_retry([t_u], batch_size=1, show_progress_bar=False)[0]
            s = self._encode_with_retry([t_s], batch_size=1, show_progress_bar=False)[0]
            base_res = s - base_u
            base_score = float((base_res @ direction) / (np.linalg.norm(base_res) * 1.0 + 1e-12))
            tokens = t_u.split()
            for j in range(len(tokens)):
                alt = " ".join(tokens[:j] + tokens[j+1:]) if len(tokens) > 1 else ""
                alt_u = self._encode_with_retry([alt], batch_size=1, show_progress_bar=False)[0]
                res = s - alt_u
                score = float((res @ direction) / (np.linalg.norm(res) * 1.0 + 1e-12))
                contrib = max(0.0, base_score - score)
                token = tokens[j].lower()
                token_contribs[token] = token_contribs.get(token, 0.0) + contrib
        if not token_contribs:
            return {}
        # Plot top tokens
        items = sorted(token_contribs.items(), key=lambda kv: kv[1], reverse=True)[:20]
        labels = [k for k, _ in items]
        vals = [v for _, v in items]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=vals, y=labels, orient='h', color='#4C78A8')
        plt.xlabel('Aggregated Contribution (Δ cosine)')
        plt.ylabel('Token')
        plt.title('Token Occlusion Attribution (top 20)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved token attribution plot to '{output_path}'")
        plt.close()
        return token_contribs

    def confound_regression(self, texts, residuals, direction):
        """
        Regress projection onto direction against simple textual confounds.
        Returns dict with R^2 and coefficients.
        """
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        y = residuals @ direction
        def features(text):
            toks = text.split()
            num_tokens = len(toks)
            avg_tok_len = np.mean([len(t) for t in toks]) if num_tokens else 0.0
            s = text
            L = len(s) + 1e-12
            counts = {
                'len_chars': len(s),
                'num_tokens': num_tokens,
                'avg_tok_len': avg_tok_len,
                'commas': s.count(','), 'semicolons': s.count(';'), 'colons': s.count(':'),
                'dashes': s.count('-'), 'exclaims': s.count('!'), 'questions': s.count('?'),
                'quotes': s.count('"') + s.count("'"), 'parens': s.count('(') + s.count(')'),
                'digits_share': sum(ch.isdigit() for ch in s) / L,
                'upper_share': sum(ch.isupper() for ch in s) / L,
            }
            return counts
        feats = [features(t) for t in texts]
        feat_names = list(feats[0].keys()) if feats else []
        X = np.array([[f[n] for n in feat_names] for f in feats]) if feats else np.zeros((len(texts), 0))
        if X.shape[1] == 0:
            return {'r2': None, 'coeffs': {}}
        lr = LinearRegression()
        lr.fit(X, y)
        y_hat = lr.predict(X)
        r2 = float(r2_score(y, y_hat))
        coeffs = {name: float(coef) for name, coef in zip(feat_names, lr.coef_)}
        return {'r2': r2, 'coeffs': coeffs}

    def visualize_extended(self, residuals, directions, topk_sims=None, svd_info=None, output_path='style_residuals_analysis_ext.png'):
        """
        Extended figure: SVD spectrum, cumulative, angles, neighbor sims, energy along dir.
        directions: dict {name: vector}
        """
        ncols = 3
        fig, axes = plt.subplots(2, ncols, figsize=(16, 10))
        # SVD spectrum
        if svd_info is not None:
            s = svd_info['singular_values']
            explained = svd_info['explained']
            axes[0, 0].plot(s, marker='o')
            axes[0, 0].set_title('Singular Values')
            axes[1, 0].plot(np.cumsum(explained))
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].set_title('Cumulative Explained Variance')
        else:
            axes[0, 0].text(0.5, 0.5, 'No SVD', ha='center', va='center')
            axes[1, 0].text(0.5, 0.5, 'No SVD', ha='center', va='center')
        # Angles
        angle_map = self.angle_diagnostics(residuals, directions)
        for name, angles in angle_map.items():
            axes[0, 1].hist(angles, bins=40, alpha=0.6, label=name)
        axes[0, 1].set_title('Angles to Directions (deg)')
        axes[0, 1].legend()
        # Neighbor sims
        if topk_sims is not None and topk_sims.size:
            axes[1, 1].hist(topk_sims.flatten(), bins=40, color='#54a24b', alpha=0.8)
            axes[1, 1].set_title('Top-k Neighbor Cosine Sims')
        else:
            axes[1, 1].text(0.5, 0.5, 'No kNN sims', ha='center', va='center')
        # Energy along first direction
        first_name = next(iter(directions))
        d = directions[first_name]
        d = d / (np.linalg.norm(d) + 1e-12)
        proj = (residuals @ d)
        axes[0, 2].hist(proj, bins=40, color='#F58518', alpha=0.85)
        axes[0, 2].set_title(f'Projection on {first_name}')
        # Residual norms
        norms = np.linalg.norm(residuals, axis=1)
        axes[1, 2].hist(norms, bins=40, color='#4C78A8', alpha=0.85)
        axes[1, 2].set_title('Residual Norms')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved extended visualization to '{output_path}'")
        plt.close()

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

        #compute cosine sims to avg (vectorized)
        avg_vec = residuals.mean(axis=0)
        cos_sims = (residuals @ avg_vec) / (
            np.clip(np.linalg.norm(residuals, axis=1), 1e-12, None) * (np.linalg.norm(avg_vec) + 1e-12)
        )

        #top-left: Cosine Similarity Heatmap (full or sampled)
        if similarity_matrix is not None and len(similarity_matrix) <= self.max_heatmap_sample_size:
            sns.heatmap(
                similarity_matrix,
                annot=False,
                cmap='coolwarm', center=0.5, vmin=0, vmax=1,
                xticklabels=False, yticklabels=False,
                ax=axes[0, 0], cbar_kws={'label': 'Cosine Similarity'}
            )
            axes[0, 0].set_title(
                'Pairwise Cosine Similarity (full)', fontsize=12, fontweight='bold'
            )
        else:
            #sample for heatmap
            m = min(self.max_heatmap_sample_size, num_samples)
            idx = self._stratified_sample_indices(residual_norms, m)
            res_norm = normalize(residuals[idx], axis=1)
            sim_sample = res_norm @ res_norm.T
            sns.heatmap(
                sim_sample, annot=False, cmap='coolwarm', center=0.5, vmin=0, vmax=1,
                xticklabels=False, yticklabels=False, ax=axes[0, 0],
                cbar_kws={'label': 'Cosine Similarity'}
            )
            axes[0, 0].set_title(
                f'Pairwise Cosine Similarity (sampled n={m})', fontsize=12, fontweight='bold'
            )

        #top-right: PCA 2D view (scatter or hexbin) colored by cosine to avg
        if num_samples > 1:
            pca = PCA(n_components=2)
            residuals_pca = pca.fit_transform(residuals)
            if num_samples <= self.max_scatter_points:
                sc = axes[0, 1].scatter(
                    residuals_pca[:, 0], residuals_pca[:, 1],
                    c=cos_sims, cmap='viridis', s=20, alpha=0.7, edgecolors='none'
                )
            else:
                hb = axes[0, 1].hexbin(
                    residuals_pca[:, 0], residuals_pca[:, 1],
                    C=cos_sims, reduce_C_function=np.mean, gridsize=40, cmap='viridis'
                )
                sc = hb
            axes[0, 1].axhline(0, color='gray', linestyle='--', alpha=0.3)
            axes[0, 1].axvline(0, color='gray', linestyle='--', alpha=0.3)
            axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
            axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
            axes[0, 1].set_title('PCA of Residuals (color = cosine to avg)', fontsize=12, fontweight='bold')
            cbar = fig.colorbar(sc, ax=axes[0, 1])
            cbar.set_label('Cosine to Average Vector')

        #bottom-left: Residual magnitude distribution
        axes[1, 0].hist(residual_norms, bins=40, color='#4C78A8', alpha=0.85)
        axes[1, 0].set_xlabel('L2 Norm', fontsize=11)
        axes[1, 0].set_ylabel('Count', fontsize=11)
        axes[1, 0].set_title('Distribution of Residual Magnitudes', fontsize=12, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)

        #bottom-right: Cosine to average vector distribution
        axes[1, 1].hist(cos_sims, bins=40, color='#F58518', alpha=0.85)
        axes[1, 1].set_xlabel('Cosine Similarity', fontsize=11)
        axes[1, 1].set_ylabel('Count', fontsize=11)
        axes[1, 1].set_title('Distribution: Cosine to Average Style Vector', fontsize=12, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)

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
        
        stylized_enc, unstylized_enc, residuals = self.encode_pairs(stylized_sentences, unstylized_sentences)
        n = len(residuals)
        
        residual_norms = self.analyze_magnitudes(residuals)
        
        avg_style_vector = self.compute_average_vector(residuals)
        
        cosine_sims = self.compute_cosine_similarities(residuals, avg_style_vector)
        
        #avoid full O(N^2) for large N; matrices will be sampled in visualize
        similarity_matrix = self.compute_similarity_matrix(residuals)
        dot_product_matrix = self.compute_dot_products(residuals)
        
        #compute local structure stats without full matrix
        topk_sims = self.compute_topk_neighbor_similarities(residuals)
        topk_summary = {
            'k': topk_sims.shape[1] if topk_sims.ndim == 2 else 0,
            'mean': float(np.mean(topk_sims)) if topk_sims.size else None,
            'median': float(np.median(topk_sims)) if topk_sims.size else None,
            'std': float(np.std(topk_sims)) if topk_sims.size else None,
        }
        print("\nLocal similarity (top-k cosine) summary:", topk_summary)
        
        self.visualize(residuals, similarity_matrix, dot_product_matrix, residual_norms, 'all_semantic_search.png')
        # Extended analyses and artifacts
        probe = self.linear_probe_style_direction(stylized_enc, unstylized_enc, residuals, avg_style_vector)
        print(f"Supervised vs Avg direction cosine: {probe['cos_w_vs_avg']:.4f}")
        svd_info = self.svd_residual_analysis(residuals)
        self.plot_quiver_arrows(stylized_enc, unstylized_enc, output_path='style_flow_quiver.png')
        directions = {'avg': avg_style_vector, 'probe_w': probe['w']}
        self.visualize_extended(residuals, directions, topk_sims=topk_sims, svd_info=svd_info, output_path='style_residuals_analysis_ext.png')
        # Lexicon-conditioned effects
        lex_avg = self.lexicon_conditioned_effects(unstylized_sentences, residuals, avg_style_vector)
        lex_w = self.lexicon_conditioned_effects(unstylized_sentences, residuals, probe['w'])
        print("Lexicon-conditioned differences (avg):", lex_avg)
        print("Lexicon-conditioned differences (w):", lex_w)
        # Token occlusion attribution (sampled)
        _ = self.token_occlusion_attribution(stylized_sentences, unstylized_sentences, probe['w'], num_pairs=None, output_path='token_attribution.png')
        # Confound regression
        confounds = self.confound_regression(unstylized_sentences, residuals, probe['w'])
        print("Confound regression:", confounds)
        
        return {
            'residuals': residuals,
            'residual_norms': residual_norms,
            'avg_style_vector': avg_style_vector,
            'cosine_sims': cosine_sims,
            'similarity_matrix': similarity_matrix,
            'dot_product_matrix': dot_product_matrix,
            'topk_sims': topk_sims,
            'topk_summary': topk_summary,
            'stylized_enc': stylized_enc,
            'unstylized_enc': unstylized_enc,
            'probe': probe,
            'svd_info': svd_info,
            'lexicon_avg': lex_avg,
            'lexicon_w': lex_w,
            'confounds': confounds,
        }

    def save_results_summary(self, results_dict, output_path='results.txt'):
        """
        Save a concise, scalable summary of results to a text file.
        """
        residuals = results_dict.get('residuals')
        residual_norms = results_dict.get('residual_norms')
        cosine_sims = results_dict.get('cosine_sims')
        topk_summary = results_dict.get('topk_summary')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Stylized Residuals Analysis Summary\n")
            f.write("="*40 + "\n\n")
            if residuals is not None:
                f.write(f"num_residuals: {len(residuals)}\n")
                f.write(f"residual_dim: {residuals.shape[1]}\n\n")
            if residual_norms is not None and len(residual_norms):
                q = np.quantile(residual_norms, [0.0, 0.25, 0.5, 0.75, 1.0])
                f.write("Residual Norms\n")
                f.write(
                    f"mean={residual_norms.mean():.6f} std={residual_norms.std():.6f} "
                    f"min={q[0]:.6f} q25={q[1]:.6f} median={q[2]:.6f} q75={q[3]:.6f} max={q[4]:.6f}\n\n"
                )
            if cosine_sims is not None and len(cosine_sims):
                q = np.quantile(cosine_sims, [0.0, 0.25, 0.5, 0.75, 1.0])
                f.write("Cosine to Average Vector\n")
                f.write(
                    f"mean={cosine_sims.mean():.6f} std={cosine_sims.std():.6f} "
                    f"min={q[0]:.6f} q25={q[1]:.6f} median={q[2]:.6f} q75={q[3]:.6f} max={q[4]:.6f}\n\n"
                )
            if topk_summary is not None:
                f.write("Top-k Neighbor Cosine Similarities\n")
                f.write(
                    f"k={topk_summary.get('k')} mean={topk_summary.get('mean')} "
                    f"median={topk_summary.get('median')} std={topk_summary.get('std')}\n"
                )
        print(f"Saved analysis summary to {output_path}")


if __name__ == "__main__":
    torch.cuda.empty_cache()

    analyzer = StylizedResiduals(
        original_text_path='data/bronte/withering_heights_merged.csv',
        output_data_path='data/bronte/withering_heights_stripped_checkpoint.csv', 
        stripped_text_path='data/bronte/withering_heights_stripped_checkpoint.csv'
    )

    stylized_df = pd.read_csv(analyzer.original_text_path)
    unstyled_df = pd.read_csv(analyzer.stripped_text_path)

    stylized_list = unstyled_df["original"].tolist()
    unstyled_list = unstyled_df["unstylized"].tolist()

    results = analyzer.run_full_analysis(stylized_list, unstyled_list)
    analyzer.save_results_summary(results, 'results.txt')


    
