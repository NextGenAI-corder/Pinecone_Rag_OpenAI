import { fetch } from 'wix-fetch';
import wixSecretsBackend from 'wix-secrets-backend';
import { webMethod, Permissions } from 'wix-web-module';

export const askOpenAI = webMethod(Permissions.Anyone, async (question) => {
  const openaiKey = await wixSecretsBackend.getSecret("OPENAI_API_KEY");
  const pineconeKey = await wixSecretsBackend.getSecret("PINECONE_API_KEY");

  const embeddingRes = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${openaiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      input: question,
      model: "text-embedding-3-small"
    })
  });

  const embedding = (await embeddingRes.json()).data[0].embedding;

  const pineconeRes = await fetch("https://YOUR-INDEX.svc.YOUR-REGION.pinecone.io/query", {
    method: "POST",
    headers: {
      "Api-Key": pineconeKey,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      topK: 3,
      includeMetadata: true,
      vector: embedding,
      namespace: "nextgen-specs"
    })
  });

  const matches = (await pineconeRes.json()).matches;
  const context = matches.map(m => m.metadata.text).join("\n");

  const chatRes = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${openaiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "gpt-4o",
      messages: [
        { role: "system", content: "You are a helpful assistant. Use only the provided context." },
        { role: "user", content: `Context:\n${context}\n\nQuestion:\n${question}` }
      ]
    })
  });

  const result = await chatRes.json();
  return result.choices[0].message.content;
});
