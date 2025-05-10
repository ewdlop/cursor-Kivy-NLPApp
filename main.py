import spacy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.core.window import Window
from textblob import TextBlob
from langdetect import detect
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import textstat
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

class NLPApp(App):
    def build(self):
        # 下载必要的 NLTK 数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # 加载 spaCy 模型
        self.nlp = spacy.load("en_core_web_sm")
        
        # 创建主布局
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # 创建输入框
        self.input_text = TextInput(
            multiline=True,
            hint_text='Enter text to analyze...',
            size_hint_y=0.3
        )
        
        # 创建分析类型选择器
        self.analysis_type = Spinner(
            text='Select Analysis Type',
            values=('Basic Analysis', 'Sentiment Analysis', 'Keyword Extraction', 
                   'Text Summary', 'Language Detection', 'Text Similarity',
                   'Text Statistics', 'Readability Analysis', 'Entity Relations',
                   'Text Classification'),
            size_hint_y=0.1
        )
        
        # 创建第二个输入框（用于相似度分析）
        self.second_input = TextInput(
            multiline=True,
            hint_text='Enter second text for similarity comparison...',
            size_hint_y=0.3,
            opacity=0
        )
        
        # 创建输出文本框
        self.output_text = TextInput(
            multiline=True,
            readonly=True,
            background_color=(0.95, 0.95, 0.95, 1),
            foreground_color=(0, 0, 0, 1),
            size_hint_y=0.4
        )
        
        # 创建按钮
        analyze_button = Button(
            text='Analyze Text',
            size_hint_y=0.1,
            on_press=self.analyze_text
        )
        
        # 添加组件到布局
        layout.add_widget(Label(text='NLP Text Analysis Tool', size_hint_y=0.1))
        layout.add_widget(self.input_text)
        layout.add_widget(self.analysis_type)
        layout.add_widget(self.second_input)
        layout.add_widget(analyze_button)
        layout.add_widget(self.output_text)
        
        # 绑定分析类型改变事件
        self.analysis_type.bind(text=self.on_analysis_type_change)
        
        return layout
    
    def on_analysis_type_change(self, instance, value):
        # 当选择相似度分析时显示第二个输入框
        if value == 'Text Similarity':
            self.second_input.opacity = 1
        else:
            self.second_input.opacity = 0
    
    def analyze_text(self, instance):
        text = self.input_text.text
        if not text:
            self.output_text.text = 'Please enter some text to analyze'
            return
        
        analysis_type = self.analysis_type.text
        
        if analysis_type == 'Basic Analysis':
            self.basic_analysis(text)
        elif analysis_type == 'Sentiment Analysis':
            self.sentiment_analysis(text)
        elif analysis_type == 'Keyword Extraction':
            self.keyword_extraction(text)
        elif analysis_type == 'Text Summary':
            self.text_summary(text)
        elif analysis_type == 'Language Detection':
            self.language_detection(text)
        elif analysis_type == 'Text Similarity':
            self.text_similarity(text)
        elif analysis_type == 'Text Statistics':
            self.text_statistics(text)
        elif analysis_type == 'Readability Analysis':
            self.readability_analysis(text)
        elif analysis_type == 'Entity Relations':
            self.entity_relations(text)
        elif analysis_type == 'Text Classification':
            self.text_classification(text)
    
    def basic_analysis(self, text):
        # 使用 spaCy 进行文本分析
        doc = self.nlp(text)
        
        # 收集分析结果
        results = []
        
        # 词性标注
        results.append("Part of Speech Tagging:")
        for token in doc:
            results.append(f"{token.text}: {token.pos_}")
        
        # 命名实体识别
        results.append("\nNamed Entities:")
        for ent in doc.ents:
            results.append(f"{ent.text}: {ent.label_}")
        
        # 依存句法分析
        results.append("\nDependency Parsing:")
        for token in doc:
            results.append(f"{token.text} -> {token.head.text} ({token.dep_})")
        
        self.output_text.text = '\n'.join(results)
    
    def sentiment_analysis(self, text):
        blob = TextBlob(text)
        results = []
        
        # 整体情感分析
        results.append("Overall Sentiment Analysis:")
        results.append(f"Polarity: {blob.sentiment.polarity:.2f} (-1 to 1, where 1 is positive)")
        results.append(f"Subjectivity: {blob.sentiment.subjectivity:.2f} (0 to 1, where 1 is subjective)")
        
        # 句子级别情感分析
        results.append("\nSentence-level Analysis:")
        for sentence in blob.sentences:
            results.append(f"\nSentence: {sentence}")
            results.append(f"Polarity: {sentence.sentiment.polarity:.2f}")
            results.append(f"Subjectivity: {sentence.sentiment.subjectivity:.2f}")
        
        self.output_text.text = '\n'.join(results)
    
    def keyword_extraction(self, text):
        # 分词
        tokens = word_tokenize(text.lower())
        
        # 移除停用词和标点符号
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
        
        # 计算词频
        fdist = FreqDist(tokens)
        
        # 使用 TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        
        results = []
        results.append("Keyword Analysis:")
        
        # 显示词频统计
        results.append("\nWord Frequency:")
        for word, freq in fdist.most_common(10):
            results.append(f"{word}: {freq}")
        
        # 显示 TF-IDF 结果
        results.append("\nTF-IDF Keywords:")
        tfidf_scores = tfidf_matrix.toarray()[0]
        top_indices = tfidf_scores.argsort()[-10:][::-1]
        for idx in top_indices:
            results.append(f"{feature_names[idx]}: {tfidf_scores[idx]:.3f}")
        
        self.output_text.text = '\n'.join(results)
    
    def text_summary(self, text):
        # 简单的基于句子的摘要
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # 计算句子重要性（基于词频）
        word_frequencies = {}
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word not in stopwords.words('english'):
                    if word not in word_frequencies:
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1
        
        # 计算句子分数
        sentence_scores = {}
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word in word_frequencies:
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_frequencies[word]
                    else:
                        sentence_scores[sentence] += word_frequencies[word]
        
        # 选择得分最高的句子
        summary_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        summary = ' '.join([sentence for sentence, score in summary_sentences])
        
        results = []
        results.append("Text Summary:")
        results.append("\nOriginal Text:")
        results.append(text)
        results.append("\nSummary:")
        results.append(summary)
        
        self.output_text.text = '\n'.join(results)
    
    def language_detection(self, text):
        try:
            lang = detect(text)
            results = []
            results.append("Language Detection:")
            results.append(f"Detected Language: {lang}")
            
            # 添加语言名称映射
            lang_names = {
                'en': 'English',
                'zh': 'Chinese',
                'es': 'Spanish',
                'fr': 'French',
                'de': 'German',
                'ja': 'Japanese',
                'ko': 'Korean',
                'ru': 'Russian'
            }
            
            if lang in lang_names:
                results.append(f"Language Name: {lang_names[lang]}")
            
            self.output_text.text = '\n'.join(results)
        except:
            self.output_text.text = "Error: Could not detect language"
    
    def text_similarity(self, text):
        if not self.second_input.text:
            self.output_text.text = "Please enter a second text for comparison"
            return
        
        # 使用 spaCy 计算相似度
        doc1 = self.nlp(text)
        doc2 = self.nlp(self.second_input.text)
        
        similarity = doc1.similarity(doc2)
        
        results = []
        results.append("Text Similarity Analysis:")
        results.append(f"\nSimilarity Score: {similarity:.2f} (0 to 1, where 1 is identical)")
        results.append("\nText 1:")
        results.append(text)
        results.append("\nText 2:")
        results.append(self.second_input.text)
        
        self.output_text.text = '\n'.join(results)
    
    def text_statistics(self, text):
        # 基本统计信息
        words = word_tokenize(text)
        sentences = nltk.sent_tokenize(text)
        
        # 计算各种统计指标
        stats = {
            'Characters': len(text),
            'Words': len(words),
            'Sentences': len(sentences),
            'Average Word Length': sum(len(word) for word in words) / len(words),
            'Average Sentence Length': len(words) / len(sentences),
            'Unique Words': len(set(words)),
            'Vocabulary Richness': len(set(words)) / len(words)
        }
        
        # 词性分布
        doc = self.nlp(text)
        pos_counts = Counter(token.pos_ for token in doc)
        
        results = []
        results.append("Text Statistics:")
        results.append("\nBasic Statistics:")
        for key, value in stats.items():
            results.append(f"{key}: {value:.2f}")
        
        results.append("\nPart of Speech Distribution:")
        for pos, count in pos_counts.most_common():
            results.append(f"{pos}: {count}")
        
        self.output_text.text = '\n'.join(results)
    
    def readability_analysis(self, text):
        # 计算各种可读性指标
        readability_scores = {
            'Flesch Reading Ease': textstat.flesch_reading_ease(text),
            'Flesch-Kincaid Grade': textstat.flesch_kincaid_grade(text),
            'Gunning Fog': textstat.gunning_fog(text),
            'SMOG Index': textstat.smog_index(text),
            'Automated Readability Index': textstat.automated_readability_index(text),
            'Coleman-Liau Index': textstat.coleman_liau_index(text),
            'Linsear Write Formula': textstat.linsear_write_formula(text),
            'Dale-Chall Readability Score': textstat.dale_chall_readability_score(text)
        }
        
        results = []
        results.append("Readability Analysis:")
        results.append("\nReadability Scores:")
        for metric, score in readability_scores.items():
            results.append(f"{metric}: {score:.2f}")
        
        # 添加解释
        results.append("\nInterpretation:")
        results.append("Flesch Reading Ease:")
        results.append("90-100: Very easy (5th grade)")
        results.append("80-89: Easy (6th grade)")
        results.append("70-79: Fairly easy (7th grade)")
        results.append("60-69: Standard (8th-9th grade)")
        results.append("50-59: Fairly difficult (10th-12th grade)")
        results.append("30-49: Difficult (College)")
        results.append("0-29: Very difficult (College graduate)")
        
        self.output_text.text = '\n'.join(results)
    
    def entity_relations(self, text):
        doc = self.nlp(text)
        
        # 创建实体关系图
        G = nx.Graph()
        entity_pairs = []
        
        # 提取实体关系
        for ent1 in doc.ents:
            for ent2 in doc.ents:
                if ent1 != ent2:
                    # 检查实体是否在同一个句子中
                    if ent1.sent == ent2.sent:
                        entity_pairs.append((ent1.text, ent2.text, ent1.label_, ent2.label_))
                        G.add_edge(ent1.text, ent2.text)
        
        results = []
        results.append("Entity Relations Analysis:")
        
        # 显示实体关系
        results.append("\nEntity Pairs in Same Sentences:")
        for ent1, ent2, type1, type2 in entity_pairs:
            results.append(f"{ent1} ({type1}) <-> {ent2} ({type2})")
        
        # 显示实体关系网络统计
        if G.nodes():
            results.append("\nEntity Network Statistics:")
            results.append(f"Number of Entities: {len(G.nodes())}")
            results.append(f"Number of Relations: {len(G.edges())}")
            results.append(f"Average Degree: {sum(dict(G.degree()).values()) / len(G.nodes()):.2f}")
        
        self.output_text.text = '\n'.join(results)
    
    def text_classification(self, text):
        # 简单的文本分类（基于关键词）
        categories = {
            'Technology': ['computer', 'software', 'hardware', 'internet', 'digital', 'data', 'system'],
            'Business': ['business', 'market', 'company', 'finance', 'investment', 'stock', 'trade'],
            'Science': ['science', 'research', 'experiment', 'study', 'scientific', 'discovery'],
            'Health': ['health', 'medical', 'disease', 'treatment', 'patient', 'doctor', 'hospital'],
            'Education': ['education', 'school', 'student', 'teacher', 'learn', 'study', 'class']
        }
        
        # 计算每个类别的得分
        scores = {}
        words = set(word_tokenize(text.lower()))
        
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in words)
            scores[category] = score
        
        # 获取最高分的类别
        max_score = max(scores.values())
        if max_score > 0:
            predicted_categories = [cat for cat, score in scores.items() if score == max_score]
        else:
            predicted_categories = ['Unknown']
        
        results = []
        results.append("Text Classification:")
        results.append("\nPredicted Categories:")
        for category in predicted_categories:
            results.append(f"- {category}")
        
        results.append("\nCategory Scores:")
        for category, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            results.append(f"{category}: {score}")
        
        self.output_text.text = '\n'.join(results)

if __name__ == '__main__':
    NLPApp().run() 