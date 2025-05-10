import spacy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.core.window import Window

class NLPApp(App):
    def build(self):
        # 加载 spaCy 模型
        self.nlp = spacy.load("en_core_web_sm")
        
        # 创建主布局
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # 创建输入框
        self.input_text = TextInput(
            multiline=True,
            hint_text='Enter text to analyze...',
            size_hint_y=0.4
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
        layout.add_widget(analyze_button)
        layout.add_widget(self.output_text)
        
        return layout
    
    def analyze_text(self, instance):
        text = self.input_text.text
        if not text:
            self.output_text.text = 'Please enter some text to analyze'
            return
        
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
        
        # 更新输出
        self.output_text.text = '\n'.join(results)

if __name__ == '__main__':
    NLPApp().run() 