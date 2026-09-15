from rag import MedicalRAG

def main():
    print("="*60)
    print("MEDICAL KNOWLEDGE INGESTION")
    print("="*60)
    try:
        rag = MedicalRAG()
        rag.create_vector_store()
        print("\nINGESTION COMPLETED SUCCESSFULLY")
        print("Vector database files created inside: vectorstore/")
        print("Run: streamlit run app.py")
    except Exception as e:
        print("\nERROR DURING INGESTION")
        print(str(e))

if __name__ == "__main__":
    main()
