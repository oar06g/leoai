
# AdvancedAI.py - نموذج ذكاء اصطناعي متقدم للتحليل غير الخاضع للإشراف

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, ViTFeatureExtractor, ViTModel
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
# تجنب استخدام UMAP بسبب مشاكل التوافق المحتملة
# from umap import UMAP
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pytesseract
import fitz  # PyMuPDF
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# استخدام مكتبات أبسط بدلاً من gensim
# from gensim.models import Word2Vec, Doc2Vec
# from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer
# تجنب استخدام spacy لتبسيط التبعيات
# import spacy

# تنزيل الموارد اللازمة
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class AdvancedAI:
    """
    نموذج ذكاء اصطناعي متقدم للتحليل غير الخاضع للإشراف والقادر على معالجة أنواع مختلفة من البيانات
    """
    
    def __init__(self, language_model="sentence-transformers/all-mpnet-base-v2", 
                 vision_model="google/vit-base-patch16-224", device=None):
        """
        تهيئة النموذج مع المكونات المختلفة
        
        Args:
            language_model: اسم نموذج اللغة المستخدم
            vision_model: اسم نموذج الرؤية المستخدم
            device: الجهاز المستخدم للحسابات (CPU/GPU)
        """
        # تحديد جهاز الحوسبة
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # تهيئة مكونات معالجة اللغة الطبيعية
        # استخدام معالجة نصوص مبسطة بدل spaCy
        self.sentence_transformer = SentenceTransformer(language_model)
        self.sentence_transformer = self.sentence_transformer.to(self.device)
        
        # تهيئة مكونات معالجة الصور
        self.vision_feature_extractor = ViTFeatureExtractor.from_pretrained(vision_model)
        self.vision_model = ViTModel.from_pretrained(vision_model)
        self.vision_model = self.vision_model.to(self.device)
        
        # إعداد أدوات التعلم غير الخاضع للإشراف
        self.pca = PCA(n_components=50)
        # استخدام t-SNE من sklearn بدلاً من UMAP
        # self.umap = UMAP(n_neighbors=15, n_components=2, min_dist=0.1)
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        
        # لتحليل النصوص
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000)
        self.count_vectorizer = CountVectorizer(max_features=10000)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # ذاكرة للسياق
        self.context_memory = []
        self.max_memory_size = 100
        
        print("نموذج الذكاء الاصطناعي المتقدم جاهز للاستخدام!")
        
    # ------------- معالجة المستندات والنصوص -------------
    
    def preprocess_text(self, text):
        """
        معالجة أولية للنصوص
        """
        # تحويل إلى أحرف صغيرة
        text = text.lower()
        
        # إزالة الأحرف الخاصة
        text = re.sub(r'[^\w\s]', '', text)
        
        # تقسيم إلى كلمات
        tokens = word_tokenize(text)
        
        # إزالة الكلمات الشائعة (stop words)
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # lemmatization
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return tokens
    
    def extract_text_from_pdf(self, pdf_path):
        """
        استخراج النص من ملف PDF
        """
        document = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
            
        return text
    
    def analyze_document(self, document_path):
        """
        تحليل وثيقة (PDF أو نص) واستخراج المعلومات المهمة
        """
        # تحديد نوع الملف واستخراج النص
        if document_path.endswith('.pdf'):
            text = self.extract_text_from_pdf(document_path)
        else:  # فرضية أنه ملف نصي
            with open(document_path, 'r', encoding='utf-8') as file:
                text = file.read()
        
        # تقسيم النص إلى جمل
        sentences = sent_tokenize(text)
        
        # استخراج الكلمات المفتاحية باستخدام TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # الحصول على أهم الكلمات المفتاحية
        scores = zip(feature_names, np.asarray(tfidf_matrix.sum(axis=0)).ravel())
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        keywords = [item[0] for item in sorted_scores[:20]]  # أهم 20 كلمة
        
        # تحليل الموضوعات باستخدام embeddings وتقنيات التجميع
        sentence_embeddings = self.sentence_transformer.encode(sentences)
        reduced_embeddings = self.pca.fit_transform(sentence_embeddings)
        clusters = self.kmeans.fit_predict(reduced_embeddings)
        
        # تلخيص الوثيقة باختيار جملة تمثيلية من كل مجموعة
        summary_sentences = []
        for cluster_id in np.unique(clusters):
            cluster_sentences = [sentences[i] for i in range(len(sentences)) if clusters[i] == cluster_id]
            if cluster_sentences:
                # اختيار الجملة الأقرب لمركز المجموعة
                cluster_center = np.mean([sentence_embeddings[i] for i in range(len(sentences)) if clusters[i] == cluster_id], axis=0)
                distances = [np.linalg.norm(sentence_embeddings[i] - cluster_center) for i in range(len(sentences)) if clusters[i] == cluster_id]
                closest_idx = np.argmin(distances)
                summary_sentences.append(cluster_sentences[closest_idx])
        
        # إعداد النتائج
        results = {
            "document_length": len(text),
            "sentences_count": len(sentences),
            "keywords": keywords,
            "summary": " ".join(summary_sentences),
            "clusters_count": len(np.unique(clusters)),
            "main_topics": self._extract_topics_from_clusters(sentences, clusters)
        }
        
        return results
    
    def _extract_topics_from_clusters(self, sentences, clusters):
        """
        استخراج الموضوعات الرئيسية من التجميعات
        """
        topics = []
        for cluster_id in np.unique(clusters):
            cluster_sentences = [sentences[i] for i in range(len(sentences)) if clusters[i] == cluster_id]
            
            # تجميع الكلمات من جمل المجموعة
            cluster_text = " ".join(cluster_sentences)
            words = self.preprocess_text(cluster_text)
            
            # حساب تكرار الكلمات
            word_freq = {}
            for word in words:
                if len(word) > 3:  # تجاهل الكلمات القصيرة
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # استخراج أهم الكلمات
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            top_words = [word for word, freq in sorted_words[:5]]
            
            topics.append(top_words)
        
        return topics
    
    # ------------- معالجة الصور -------------
    
    def analyze_image(self, image_path):
        """
        تحليل صورة واستخراج المعلومات منها
        """
        # قراءة الصورة
        image = Image.open(image_path)
        
        # استخراج النص من الصورة باستخدام OCR
        img_cv = cv2.imread(image_path)
        text = pytesseract.image_to_string(img_cv)
        
        # استخراج features من الصورة باستخدام نموذج الرؤية
        inputs = self.vision_feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vision_model(**inputs)
            image_features = outputs.last_hidden_state[:, 0].cpu().numpy()  # استخدام CLS token
        
        # تحليل الصورة باستخدام OpenCV
        img_height, img_width = img_cv.shape[:2]
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # تحليل الألوان الرئيسية
        pixels = np.float32(img_cv.reshape(-1, 3))
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        _, counts = np.unique(labels, return_counts=True)
        
        # تحويل الألوان إلى قيم RGB
        palette = palette.astype(int)
        dominant_colors = palette[np.argsort(counts)[::-1]]
        
        # إعداد النتائج
        results = {
            "image_dimensions": (img_width, img_height),
            "extracted_text": text,
            "contours_count": len(contours),
            "dominant_colors": dominant_colors.tolist(),
            "image_features": image_features.tolist()
        }
        
        return results
    
    # ------------- معالجة البيانات الهيكلية -------------
    
    def analyze_structured_data(self, data_path):
        """
        تحليل البيانات الهيكلية (CSV، Excel)
        """
        # قراءة البيانات
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(data_path)
        else:
            raise ValueError("صيغة الملف غير مدعومة. الصيغ المدعومة: CSV، XLS، XLSX")
        
        # معلومات عامة عن البيانات
        basic_info = {
            "rows_count": df.shape[0],
            "columns_count": df.shape[1],
            "columns": df.columns.tolist(),
            "data_types": {col: str(df[col].dtype) for col in df.columns},
            "missing_values": df.isnull().sum().to_dict()
        }
        
        # تحليل إحصائي
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        statistical_analysis = {}
        
        if not numerical_cols.empty:
            statistical_analysis = {
                "descriptive_stats": df[numerical_cols].describe().to_dict(),
                "correlations": df[numerical_cols].corr().to_dict() if len(numerical_cols) > 1 else {}
            }
        
        # اكتشاف الأنماط والتجميعات في البيانات الرقمية
        patterns = {}
        if len(numerical_cols) >= 2:
            # استخدام تقليل الأبعاد لتصور البيانات
            numeric_data = df[numerical_cols].fillna(df[numerical_cols].mean())
            if numeric_data.shape[0] > 1:  # تأكد من وجود أكثر من صف
                # PCA
                pca_result = PCA(n_components=min(2, len(numerical_cols))).fit_transform(numeric_data)
                
                # تطبيق K-means
                if numeric_data.shape[0] >= 5:  # على الأقل 5 صفوف لتطبيق التجميع
                    n_clusters = min(5, numeric_data.shape[0] - 1)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(numeric_data)
                    
                    # توزيع البيانات على المجموعات
                    cluster_distribution = pd.Series(clusters).value_counts().to_dict()
                    
                    patterns = {
                        "pca_components": pca_result.tolist(),
                        "clusters": clusters.tolist(),
                        "cluster_distribution": cluster_distribution
                    }
        
        # اكتشاف القيم الشاذة
        outliers = {}
        for col in numerical_cols:
            if df[col].nunique() > 1:  # تأكد من وجود أكثر من قيمة فريدة
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        
        # تحليل العلاقات بين الأعمدة
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        relationships = {}
        
        if not categorical_cols.empty and not numerical_cols.empty:
            for cat_col in categorical_cols[:3]:  # تحليل أول 3 أعمدة فقط لتجنب الحسابات الكثيرة
                for num_col in numerical_cols[:3]:
                    if df[cat_col].nunique() <= 10:  # فقط إذا كان عدد القيم المختلفة معقولاً
                        group_means = df.groupby(cat_col)[num_col].mean().to_dict()
                        relationships[f"{cat_col}_vs_{num_col}"] = group_means
        
        # تجميع النتائج
        results = {
            "basic_info": basic_info,
            "statistical_analysis": statistical_analysis,
            "patterns": patterns,
            "outliers": outliers,
            "relationships": relationships
        }
        
        return results
    
    # ------------- المكونات الرئيسية للتحليل غير الخاضع للإشراف -------------
    
    def unsupervised_analysis(self, data, analysis_type="text"):
        """
        تحليل غير خاضع للإشراف للبيانات
        
        Args:
            data: البيانات المراد تحليلها (نص، مسار ملف صورة، DataFrame)
            analysis_type: نوع التحليل ('text', 'image', 'structured')
        
        Returns:
            نتائج التحليل
        """
        if analysis_type == "text":
            # تحويل النص إلى تمثيل رقمي
            if isinstance(data, str):
                sentences = sent_tokenize(data)
                embeddings = self.sentence_transformer.encode(sentences)
                
                # تطبيق تقنيات التجميع وتقليل الأبعاد
                reduced_embeddings = self.pca.fit_transform(embeddings)
                # استخدام PCA بدلاً من UMAP
                # umap_embeddings = self.umap.fit_transform(embeddings)
                clusters = self.kmeans.fit_predict(reduced_embeddings)
                
                # استخراج الموضوعات
                topics = self._extract_topics_from_clusters(sentences, clusters)
                
                return {
                    "sentences_count": len(sentences),
                    "topics": topics,
                    "clusters": clusters.tolist(),
                    "visualizations": {
                        "pca": reduced_embeddings[:, :2].tolist()  # استخدام أول مكونين من PCA للتصور
                    }
                }
            else:
                raise ValueError("النص يجب أن يكون سلسلة من الأحرف (string)")
                
        elif analysis_type == "image":
            # تحليل صورة
            if isinstance(data, str) and os.path.isfile(data):
                return self.analyze_image(data)
            else:
                raise ValueError("يجب توفير مسار صالح للصورة")
                
        elif analysis_type == "structured":
            # تحليل بيانات هيكلية
            if isinstance(data, pd.DataFrame):
                # تحليل DataFrame مباشرة
                basic_info = {
                    "rows_count": data.shape[0],
                    "columns_count": data.shape[1],
                    "columns": data.columns.tolist(),
                    "data_types": {col: str(data[col].dtype) for col in data.columns},
                    "missing_values": data.isnull().sum().to_dict()
                }
                
                # تطبيق تقنيات التجميع وتقليل الأبعاد على البيانات الرقمية
                numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
                
                if len(numerical_cols) >= 2:
                    numeric_data = data[numerical_cols].fillna(data[numerical_cols].mean())
                    
                    if numeric_data.shape[0] > 2:
                        pca_result = self.pca.fit_transform(numeric_data)
                        clusters = self.kmeans.fit_predict(numeric_data)
                        
                        return {
                            "basic_info": basic_info,
                            "pca_components": pca_result.tolist(),
                            "clusters": clusters.tolist()
                        }
                
                return {"basic_info": basic_info}
                
            elif isinstance(data, str) and os.path.isfile(data):
                return self.analyze_structured_data(data)
            else:
                raise ValueError("يجب توفير DataFrame أو مسار ملف صالح للبيانات الهيكلية")
        else:
            raise ValueError("نوع التحليل غير مدعوم. الأنواع المدعومة: 'text', 'image', 'structured'")
    
    # ------------- واجهة المستخدم البسيطة -------------
    
    def analyze_file(self, file_path):
        """
        تحليل ملف وتحديد نوعه ثم تطبيق التحليل المناسب
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"الملف {file_path} غير موجود")
            
        file_extension = file_path.split('.')[-1].lower()
        
        # تحديد نوع الملف
        if file_extension in ['txt', 'pdf', 'doc', 'docx']:
            print(f"تحليل مستند: {file_path}")
            return self.analyze_document(file_path)
            
        elif file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
            print(f"تحليل صورة: {file_path}")
            return self.analyze_image(file_path)
            
        elif file_extension in ['csv', 'xls', 'xlsx']:
            print(f"تحليل بيانات هيكلية: {file_path}")
            return self.analyze_structured_data(file_path)
            
        else:
            raise ValueError(f"صيغة الملف {file_extension} غير مدعومة")
    
    def update_context(self, new_context):
        """
        تحديث سياق المحادثة
        """
        self.context_memory.append(new_context)
        if len(self.context_memory) > self.max_memory_size:
            self.context_memory.pop(0)


# -------- مثال للاستخدام --------

def main():
    # إنشاء نموذج الذكاء الاصطناعي
    ai_model = AdvancedAI()
    
    # مثال لتحليل النصوص
    print("\n=== تحليل النصوص ===")
    sample_text = """
    الذكاء الاصطناعي هو محاكاة للذكاء البشري في الآلات. يشمل مجالات مثل التعلم الآلي والتعلم العميق ومعالجة اللغة الطبيعية.
    تطبيقات الذكاء الاصطناعي متنوعة، من التشخيص الطبي إلى السيارات ذاتية القيادة. يعتبر التعلم العميق من أكثر فروع الذكاء 
    الاصطناعي تقدمًا في السنوات الأخيرة. يستخدم التعلم العميق شبكات عصبية متعددة الطبقات لمعالجة البيانات المعقدة.
    """
    text_analysis = ai_model.unsupervised_analysis(sample_text, analysis_type="text")
    print(f"تم تحليل النص: اكتشاف {len(text_analysis['topics'])} موضوعات رئيسية")
    for i, topic in enumerate(text_analysis['topics']):
        print(f"الموضوع {i+1}: {', '.join(topic)}")
    
    # إظهار كيفية استخدام النموذج على مجموعة متنوعة من الملفات
    print("\n=== تحليل الملفات ===")
    print("لتحليل ملف PDF: ai_model.analyze_file('example.pdf')")
    print("لتحليل صورة: ai_model.analyze_file('example.jpg')")
    print("لتحليل ملف بيانات: ai_model.analyze_file('example.csv')")


# if __name__ == "__main__":
    # main()
