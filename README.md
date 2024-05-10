# news_classifier_model
## LLM 기반 뉴스/아티클 자동 분류 시스템
본 프로젝트는 LLM(Large Language Model) 기술을 활용하여 뉴스 기사 및 온라인 콘텐츠를 사용자 맞춤형 분류 체계에 따라 자동으로 분류하는 시스템을 구현합니다.

### 주요 기능
- LLM 파이프라인: 최신 LLM 모델을 활용하여 텍스트 입력에 대한 의미 이해 및 분류 수행
- RAG(Retrieval-Augmented Generation): 검색 시스템과 LLM을 결합하여 더 정확한 분류 결과 도출
- 벡터 데이터베이스: FAISS 등의 벡터 DB 활용으로 빠른 유사도 검색 및 추천 기능 제공
- LangChain 프레임워크: LLM 및 검색 시스템 통합을 위한 LangChain 활용
### 기대 효과
- 사용자 맞춤형 콘텐츠 추천 및 분류로 정보 접근성 향상
- 대량의 텍스트 데이터를 효율적으로 관리 및 활용
- 지속적인 모델 업데이트를 통한 분류 정확도 향상

### 향후 계획
- Re-Ranking 적용
- 벡터 데이터베이스 Pgvector 적용