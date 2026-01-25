# Deep Learning Text Summarization

CNN/DailyMail text summarization และ headline generation ด้วย BERT Encoder-Decoder

## 📋 โครงการ

สร้างโมเดล 2 ตัวจาก CNN/DailyMail dataset (287k samples):
1. **Summarization Model** - เทรนจากศูนย์ (from scratch) 10 epochs
2. **Headline Generation Model** - ใช้ pretrained BERT 5 epochs

## 🎯 วัตถุประสงค์

งานเรียนรู้เพื่อทำความเข้าใจ:
- การเทรนโมเดล Seq2Seq จากศูนย์
- ความแตกต่างระหว่าง from-scratch vs pretrained models
- ปัญหา hallucination ใน text generation
- การจัดการกับ checkpoint และ crash recovery

## 🏗️ สถาปัตยกรรม

- **Encoder**: BERT-base-uncased (12 layers, 768 hidden)
- **Decoder**: BERT-base-uncased (12 layers, 768 hidden)
- **Parameters**: ~247M parameters
- **Training**: Mixed precision (FP16) with gradient accumulation

## 📊 การเทรน

### Summarization Model (from scratch)
- Epochs: 10 (8 initial + 2 fine-tuning)
- Learning Rate: 3e-5 → 1.5e-5
- Batch Size: 4 × 4 (gradient accumulation)
- Time: ~2 hours/epoch on RTX 4080

### Headline Model (pretrained)
- Epochs: 5
- Learning Rate: 3e-5
- Pretrained: BERT-base encoder & decoder

## 🔧 Anti-Hallucination Parameters

```python
repetition_penalty=2.0
length_penalty=1.0
no_repeat_ngram_size=3
num_beams=5
```

## 📈 ผลลัพธ์

### Summarization (from scratch)
- ❌ มีปัญหา hallucination สูง
- ⚠️ Dataset 287k ไม่เพียงพอสำหรับโมเดล 247M params
- 📊 แสดงให้เห็นความยากของการเทรนจากศูนย์

### Headline Generation (pretrained)
- ✅ ผลลัพธ์ดี สามารถใช้งานได้
- ✅ Pretrained knowledge ช่วยมาก

## 🚀 การใช้งาน

1. ติดตั้ง dependencies:
```bash
pip install torch transformers datasets rouge-score
```

2. เปิด `workshop3.ipynb` และรันทีละ cell ตามลำดับ

3. ตรวจสอบ checkpoints ใน `checkpoints_new/`

## 📁 ไฟล์สำคัญ

- `workshop3.ipynb` - Notebook หลัก (18 cells)
- `my_tokenizer_287k.json` - Custom tokenizer
- `.gitignore` - ป้องกันอัพโหลดไฟล์ขนาดใหญ่

## 💡 บทเรียน

1. **Dataset matters**: 287k samples ไม่พอสำหรับ from-scratch training
2. **Pretrained wins**: Pretrained models ได้ผลดีกว่ามาก
3. **Hallucination is hard**: ต้องใช้ generation parameters และ data เยอะ
4. **Checkpoints are crucial**: ต้องมีระบบกู้คืนเมื่อคอมดับ

## 🎓 สรุป

โปรเจกต์นี้เป็นงานเรียนรู้ที่แสดงให้เห็น:
- ข้อจำกัดของการเทรนจากศูนย์
- ความสำคัญของ pretrained models
- วิธีจัดการกับปัญหาจริงในการเทรน (crash, hallucination)

สำหรับงานจริง แนะนำให้ใช้ pretrained models เสมอ!

## 📝 License

Educational project - free to use and learn from

## 👤 Author

Nawin01234

---

⭐ ถ้าเป็นประโยชน์ อย่าลืม star repo นี้!
