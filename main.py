
# example.py - مثال تطبيقي لاستخدام النموذج

from aiadvanced import AdvancedAI
import os

def main():
    print("Launching an advanced AI model for unsupervised analysis")
    
    # إنشاء نموذج الذكاء الاصطناعي
    ai = AdvancedAI()
    
    # مثال لتحليل نص
    sample_text = """Artificial Intelligence (AI) is a branch of computer science concerned with the study and development of computer systems and programs capable of mimicking intelligent human behavior. This includes the ability to learn, reason, make decisions, and solve complex problems.

AI includes several subfields, such as machine learning, deep learning, natural language processing, and computer vision.

Prominent applications of AI include recommendation systems, self-driving cars, virtual assistants, medical diagnostic systems, and intelligent robotics.
    """
    
    print("\n=== Text analysis ===")
    text_analysis = ai.unsupervised_analysis(sample_text, analysis_type="text")
    print("The text has been analyzed.:")
    print(f"- Number of sentences: {text_analysis['sentences_count']}")
    print(f"- Number of groups: {len(set(text_analysis['clusters']))}")
    print("\n Discovered topics:")
    for i, topic in enumerate(text_analysis['topics']):
        print(f"- the topic {i+1}: {', '.join(topic)}")
    
    # مثال لكيفية تحليل ملف
    # print("\n=== أمثلة لكيفية تحليل الملفات ===")
    # print("1. لتحليل مستند:")
    # print("   results = ai.analyze_document('example.pdf')")
    # print("   print(results['summary'])  # طباعة ملخص المستند")
    
    # print("\n2. لتحليل صورة:")
    # print("   results = ai.analyze_image('example.jpg')")
    # print("   print(results['extracted_text'])  # طباعة النص المستخرج من الصورة")
    
    # print("\n3. لتحليل بيانات:")
    # print("   results = ai.analyze_structured_data('example.csv')")
    # print("   print(results['patterns'])  # طباعة الأنماط المكتشفة")

if __name__ == "__main__":
    main()
