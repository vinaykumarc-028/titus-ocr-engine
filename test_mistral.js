const { Mistral } = require('@mistralai/mistralai');
require('dotenv').config();

const apiKey = process.env.MISTRAL_API_KEY;
console.log('Testing Mistral key:', apiKey ? 'Loaded (starts with ' + apiKey.substring(0, 4) + ')' : 'Not Loaded');

const client = new Mistral({ apiKey });

async function run() {
  try {
    const res = await client.chat.complete({
      model: 'pixtral-large-latest',
      messages: [
        { role: 'user', content: 'Say hello!' }
      ]
    });
    console.log('API Success:', res.choices[0].message.content);
  } catch (err) {
    console.error('API Error:', err);
  }
}

run();
