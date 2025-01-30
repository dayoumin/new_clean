## `app_local.py`의 주요 기능과 사용법

### **1. 주요 기능**

1. **벡터스토어 생성**

   - `documents` 디렉토리의 PDF 파일을 읽어 벡터스토어를 생성.
   - 각 페이지를 별도의 문서로 처리하고, 메타데이터(`source`, `page`, `total_pages`)를 추가.
   - 생성된 벡터스토어는 암호화되어 저장.
2. **특정 문서 제거**

   - 벡터스토어에서 특정 문서를 제거.
   - 문서 이름(`source`)을 기준으로 해당 문서의 모든 페이지를 제거.
3. **특정 문서 추가**

   - 벡터스토어에 특정 문서를 추가.
   - 문서의 각 페이지를 처리하여 벡터스토어에 병합.
4. **암호화 및 복호화**

   - 벡터스토어 파일(`index.faiss`, `index.pkl`)을 암호화하여 저장.
   - 필요 시 복호화하여 사용.

---

### **2. 사용법**

#### **1) 벡터스토어 생성**

```bash
python local/app_local.py
```

#### **2) 특정 문서 제거**

```bash
python local/app_local.py remove "문서명.pdf"
```

- 예: `공직자윤리법.pdf` 문서 제거
  ```bash
  python local/app_local.py remove "공직자윤리법.pdf"
  ```

#### **3) 특정 문서 추가**

```bash
python local/app_local.py add "문서명.pdf"
```

- 예: `부정청탁금지법.pdf` 문서 추가
  ```bash
  python local/app_local.py add "부정청탁금지법.pdf"
  ```

---

### **3. 파일 구조**

- **`documents/`**

  - 벡터스토어로 변환할 PDF 파일을 저장.
  - 파일명 형식: `문서명.pdf` (예: `민법.pdf`).
- **`faiss_vectorstore/`**

  - 생성된 벡터스토어 파일이 저장됨.
  - `index.faiss.enc`, `index.pkl.enc` (암호화된 파일).
- **`vectorstore_key.key`**

  - 벡터스토어 암호화에 사용된 키 파일.

---

### 4. 주의사항

- **파일명 형식**

  - 파일명에 페이지 번호가 포함되지 않으므로, `문서명.pdf` 형식으로 저장.
- **암호화**

  - 벡터스토어는 기본적으로 암호화되어 저장되며, 사용 시 자동으로 복호화.
- **로그 확인**

  - 실행 시 콘솔에 출력되는 로그를 확인하여 오류를 파악.

---

### **6. 예시**

#### **벡터스토어 생성**

```bash
python local/app_local.py
```

#### **문서 제거**

```bash
python local/app_local.py remove "공직자윤리법.pdf"
```

#### **문서 추가**

```bash
python local/app_local.py add "부정청탁금지법.pdf"
```
