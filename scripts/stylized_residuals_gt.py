import os
import numpy as np 
import pandas as pd 
from google.cloud import translate



class StylizedResiduals:
    """
    Testing H1 for stylx 
    """
    
    def __init__(self, original_text_path, output_data_path, stripped_text_path, gt, project_id=None):
        """
        Initialize the StylizedResiduals analyzer.
        
        Args:
            original_text_path: Path to CSV file containing stylized sentences
            output_data_path: Path to save output CSV files
            stripped_text_path: Path to text that have been stripped from their "style"
            gt: Boolean, if True use Google Translate API
            project_id: Google Cloud project ID. Defaults to GOOGLE_CLOUD_PROJECT env var.
        """
        self.google_translate = gt
        self.original_language = "en-US"
        self.target_languages = ["nl", "en-US", "fr", "en-US", "de", "en-US"]

        self.original_text_path = original_text_path
        self.output_data_path = output_data_path
        self.stripped_text_path = stripped_text_path
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if self.google_translate and not self.project_id:
            raise ValueError("A project_id is required when using Google Translate.")
        # Initialize the client once and reuse it
        self.translation_client = translate.TranslationServiceClient() if self.google_translate else None

    def translate_to_target(self, sentence, target_language, source_language = None):
        """
        Translate a sentence to a target language with self.translator and self.tokenizer 
        
        Args:
            sentence: The text to translate
            target_language: Target language name (e.g., "French")
            
        Returns:
            Translated text
        """
        location = "global"
        parent = f"projects/{self.project_id}/locations/{location}"

        response = self.translation_client.translate_text(
            request={
                "parent": parent,
                "contents": [sentence],
                "mime_type": "text/plain",
                "source_language_code": f"{source_language}",
                "target_language_code": f"{target_language}",
            }
        )
        
        # Return the translated text from the first result
        return response.translations[0].translated_text if response.translations else ""

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
            prev = self.original_language

            #translate through each target language sequentially
            for language in self.target_languages:
                print(f"Translating to {language}...")
                if not self.google_translate:
                    curr = self.translate_to_target(curr, language)
                else:
                    curr = self.translate_to_target(curr, language, prev)
                    prev = language
            
            # The final translated sentence is in 'curr' after the loop
            stripped_sentence = curr

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

if __name__ == "__main__":

    analyzer = StylizedResiduals(
        original_text_path='/home/xikron/Documents/Cornell 2028/Academic/Sophmore/Fall/CS-6784-Sun/stylx/data/withering_heights_merged.csv',
        output_data_path='/home/xikron/Documents/Cornell 2028/Academic/Sophmore/Fall/CS-6784-Sun/stylx/data/withering_heights_stripped.csv', 
        stripped_text_path='/home/xikron/Documents/Cornell 2028/Academic/Sophmore/Fall/CS-6784-Sun/stylx/data/withering_heights_stripped.csv',
        gt=True, # Set to True to use Google Translate
        project_id='stylx-476417'
    )

    stylized_df = pd.read_csv(analyzer.original_text_path)
    
    print("startting strip")
    unstylized_sentences = analyzer.strip_style(stylized_df['sentence'].tolist())

    analyzer.save_to_csv(unstylized_sentences, analyzer.stripped_text_path)

    
