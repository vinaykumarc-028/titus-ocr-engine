const { Mistral } = require('@mistralai/mistralai');
require('dotenv').config();

const apiKey = process.env.MISTRAL_API_KEY;
const client = new Mistral({ apiKey });

async function run() {
  try {
    const res = await client.chat.complete({
      model: 'pixtral-large-latest',
      messages: [
        { role: 'user', content: 'Output the number 1 to 5 as a strict JSON array under the key "numbers".' }
      ],
      responseFormat: { type: 'json_object' }
    });
    console.log('JSON Format Success:', res.choices[0].message.content);
  } catch (err) {
    console.error('JSON Format Error:', err.message || err);
  }
}

run();
