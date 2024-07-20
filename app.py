from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = FastAPI()

class Query(BaseModel):
    text: str

# Load the pre-trained model and tokenizer
model_name = "Saidtaoussi/AraT5_Darija_to_MSA"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

@app.post("/translate")
async def translate_text(query: Query):
    try:
        # Tokenize the input text
        inputs = tokenizer(query.text, return_tensors="pt", padding=True)
        # Generate translation
        translated = model.generate(**inputs)
        # Decode the translation
        output_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        return {"response": output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
