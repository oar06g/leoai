
# example.py - مثال تطبيقي لاستخدام النموذج

from .aiadvanced import AdvancedAI
import os

def main():
    print("بدء تشغيل نموذج الذكاء الاصطناعي المتقدم للتحليل غير الخاضع للإشراف")
    
    # إنشاء نموذج الذكاء الاصطناعي
    ai = AdvancedAI()
    
    # مثال لتحليل نص
    sample_text = """
    الذكاء الاصطناعي (Artificial Intelligence أو AI) هو فرع من فروع علوم الحاسب الآلي يهتم بدراسة وتطوير أنظمة وبرامج حاسوبية 
    قادرة على محاكاة السلوك البشري الذكي. يشمل ذلك القدرة على التعلم والتفكير واتخاذ القرارات وحل المشكلات المعقدة.
    
    يتضمن الذكاء الاصطناعي العديد من المجالات الفرعية مثل التعلم الآلي (Machine Learning) والتعلم العميق (Deep Learning) 
    ومعالجة اللغات الطبيعية (Natural Language Processing) والرؤية الحاسوبية (Computer Vision).
    
    من أبرز تطبيقات الذكاء الاصطناعي: أنظمة التوصية، السيارات ذاتية القيادة، المساعدين الافتراضيين، 
    أنظمة التشخيص الطبي، والروبوتات الذكية.
    """
    
    print("\n=== تحليل النص ===")
    text_analysis = ai.unsupervised_analysis(sample_text, analysis_type="text")
    print("تم تحليل النص:")
    print(f"- عدد الجمل: {text_analysis['sentences_count']}")
    print(f"- عدد المجموعات: {len(set(text_analysis['clusters']))}")
    print("\nالموضوعات المكتشفة:")
    for i, topic in enumerate(text_analysis['topics']):
        print(f"- الموضوع {i+1}: {', '.join(topic)}")
    
    # مثال لكيفية تحليل ملف
    print("\n=== أمثلة لكيفية تحليل الملفات ===")
    print("1. لتحليل مستند:")
    print("   results = ai.analyze_document('example.pdf')")
    print("   print(results['summary'])  # طباعة ملخص المستند")
    
    print("\n2. لتحليل صورة:")
    print("   results = ai.analyze_image('example.jpg')")
    print("   print(results['extracted_text'])  # طباعة النص المستخرج من الصورة")
    
    print("\n3. لتحليل بيانات:")
    print("   results = ai.analyze_structured_data('example.csv')")
    print("   print(results['patterns'])  # طباعة الأنماط المكتشفة")

if __name__ == "__main__":
    main()
