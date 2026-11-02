# YouTube Video Q&A using RAG

Finding a specific answer in a long YouTube video can take time. I built this project to make that easier: it takes a video's transcript and lets you ask questions about what was discussed.

The project uses **Retrieval-Augmented Generation (RAG)**, so the model looks for relevant parts of the transcript before generating an answer.

## How it works

1. **Get the transcript** – Fetch English captions from a YouTube video.
2. **Split the text** – Break the transcript into smaller, overlapping chunks.
3. **Create embeddings** – Convert chunks into vectors using `all-MiniLM-L6-v2` and store them in FAISS.
4. **Find relevant context** – Retrieve the four chunks most relevant to the question.
5. **Generate an answer** – Use `Qwen2.5-1.5B-Instruct` to answer based on the retrieved context.

**Workflow:** YouTube video → Transcript → Text chunks → Embeddings + FAISS → Relevant context → Qwen answer

## Tech stack

- Python and Google Colab
- LangChain
- YouTube Transcript API
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- Hugging Face Transformers (`Qwen2.5-1.5B-Instruct`)

## Run in Google Colab

1. Open [`Video_Transcript_RAG_Gemini_Colab_proj.ipynb`](Video_Transcript_RAG.ipynb) in Google Colab.
2. Select a **T4 GPU** runtime if one is available.
3. Run the package installation cells.
4. Replace `video_id` with the ID of a YouTube video that has English captions.
5. Change the `question` variable to something you want to ask, then run the remaining RAG steps in order.

For example:

```python
video_id = "Gfr50f6ZBvo"
question = "Is nuclear fusion discussed in this video?"
```

**Note:** The notebook includes an unused Gemini embeddings initialization cell. The actual FAISS index uses Hugging Face embeddings, so you can skip the `GoogleGenerativeAIEmbeddings(...)` cell if you haven't configured a Google API key.

## Current limitations

- The video needs an accessible English transcript.
- Answers depend on what the retrieved transcript sections contain.
- Running the Qwen model locally in Colab needs enough RAM or GPU memory.

This is a notebook-based project focused on understanding the basic RAG pipeline, rather than a deployed web application.
