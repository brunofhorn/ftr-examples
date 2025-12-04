import { pipeline, cos_sim } from "@huggingface/transformers";
import { GoogleGenAI } from "@google/genai";

// Text Embedder
const embedder = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2",
  { dtype: "q8" }
);

// Image Embedder
const imgEmbedder = await pipeline(
  "image-feature-extraction",
  "Xenova/clip-vit-base-patch32",
  { dtype: "q8 " }
);

async function embedText(text) {
  await embedder(text, { pooling: "mean", normalize: true }).then(
    (t) => t.tolist()[0]
  );
}

async function embedImg(img) {
  return imgEmbedder(img, { pooling: "cls", normalize: true }).then(
    (t) => t.tolist()[0]
  );
}

async function embedWithGoogle(text) {
  return genai.models
    .embedContent({
      model: "models/text-embedding-004",
      contents: [text],
    })
    .then((r) => r.embeddings[0].values);
}

// Calcula a similaridade entre as palavras (ou frases) de 0 à 1.
console.log(cos_sim(await embedText("king"), await embedText("queen")));

const babyHippo =
  "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTDL7ALMfcT1M43_gxAGj3FCCcw6QBOEyZZ4g&s";
const babyHippo2 =
  "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRcJZ0Dj3E22c0BKNDcrCQn0MtDuoT2Q5FuoQ&s";

// Calcula a similaridade entre as imagens de 0 à 1.
console.log(cos_sim(await embedImg(babyHippo), await embedImg(babyHippo2)));

const genai = new GoogleGenAI({ apiKey: process.env.GOOGLE_GEN_AI_API_KEY });

// Calcula a similaridade entre textos (ou frases) através de APIs públicas como o Gemini
console.log(cos_sim(await embedWithGoogle("olá tudo bem?"), await embedWithGoogle("Oi meu nome é Bruno")))