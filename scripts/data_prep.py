import pandas as pd
import re


class DataPrep:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None

    def load_data(self):
        """Load the entire text file into a single string."""
        with open(self.data_path, 'r', encoding='utf-8') as file:
            self.data = file.read()
        return self.data

    def preprocess_data(self):
        """Extract sentences by removing newlines and splitting on periods."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        text = self.data.replace('\n', ' ')
        
        text = re.sub(r'\s+', ' ', text)
        sentences = re.split(r'\.(?=\s+[A-Z])', text)
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        sentences = [s[:-1] if s.endswith('..') else s for s in sentences]
        
        sentences = [s for s in sentences if len(s) > 3]
        
        return sentences

    def save_to_csv(self, sentences, output_path):
        """Save sentences to a CSV file with one sentence per row."""
        df = pd.DataFrame({'sentence': sentences})
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Saved {len(sentences)} sentences to {output_path}")
        return df
    

    def merge_sentences(self, sentences, i):
        """Merge every i consecutive sentences into one row."""
        if isinstance(sentences, str):
            original_data = pd.read_csv(self.data_path)
            sentences = original_data['sentence'].tolist()
        
        merged_sentences = []
        temp = ""
        
        for idx, sentence in enumerate(sentences, 1):
            temp += sentence if not temp else " " + sentence
            
            if idx % i == 0:
                merged_sentences.append(temp)
                temp = ""
        
        if temp:
            merged_sentences.append(temp)
        
        return merged_sentences
            

if __name__ == "__main__":
    csv_path = '/home/cc2864/Documents/cs6784/kafka_sentences.csv'
    prep = DataPrep(csv_path)
    
    print("Loading sentences from CSV...")
    df = pd.read_csv(csv_path)
    sentences = df['sentence'].tolist()
    print(f"Loaded {len(sentences)} sentences")
    
    merge_count = 3
    print(f"\nMerging every {merge_count} sentences...")
    merged_sentences = prep.merge_sentences(sentences, merge_count)
    print(f"Result: {len(merged_sentences)} merged rows")
    
    print("\nFirst 3 merged sentences:")
    for i, sent in enumerate(merged_sentences[:3], 1):
        print(f"\n{i}. {sent[:200]}..." if len(sent) > 200 else f"\n{i}. {sent}")
    
    output_path = '/home/cc2864/Documents/cs6784/kafka_sentences_merged.csv'
    prep.save_to_csv(merged_sentences, output_path)