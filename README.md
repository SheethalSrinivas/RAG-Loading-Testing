# RAG-Loading-Testing
!pip -q uninstall -y google-adk langgraph-prebuilt \
  opentelemetry-sdk opentelemetry-api \
  opentelemetry-exporter-otlp-proto-http opentelemetry-exporter-otlp-proto-common opentelemetry-proto \
  opentelemetry-exporter-gcp-logging
# Install required libraries
!pip install -q langchain_community==0.3.27 \
              langchain==0.3.27 \
              chromadb==1.0.15 \
              pymupdf==1.26.3 \
              tiktoken==0.9.0 \
              datasets==4.0.0 \
              langchain_openai==0.3.30
