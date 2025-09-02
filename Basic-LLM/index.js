import { AzureOpenAI,OpenAI } from "openai";
import 'dotenv/config'

const apiKey = process.env.AZURE_API_KEY;
const apiVersion = process.env.AZURE_API_VERSION;
const endpoint = process.env.AZURE_ENDPOINT;
const modelName = "gpt-5-mini";
const deployment = "gpt-5-mini";
let options = { endpoint, apiKey, deployment, apiVersion }
let client = new AzureOpenAI(options);

/*
  // for local host
  modelName = "gemma-3-4b-it"
  deployment = "gemma-3-4b-it"
  endpoint = "http://localhost:1234/v1/";
  apiKey = "lm-studio";
  apiVersion = ""
  options = { baseURL:endpoint, apiKey, deployment, apiVersion }
  client = new OpenAI(options)
*/



export async function main() {

  const response = await client.chat.completions.create({
    messages: [
      { role:"system", content: "You are a helpful assistant." },
      { role:"user", content: "I am going to Paris, what should I see?" }
    ],
    max_completion_tokens: 100000,
      model: modelName
  });

  if (response?.error !== undefined && response.status !== "200") {
    throw response.error;
  }
  console.log(response.choices[0].message.content);
}

main().catch((err) => {
  console.error("The sample encountered an error:", err);
});