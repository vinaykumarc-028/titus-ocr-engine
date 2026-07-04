const { Mistral } = require('@mistralai/mistralai');
require('dotenv').config();

const apiKey = process.env.MISTRAL_API_KEY;
const client = new Mistral({ apiKey });

// Small 1x1 black PNG base64
const mockBase64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';

async function testWithImageUrlKey() {
  console.log('--- Testing with imageUrl key ---');
  try {
    const res = await client.chat.complete({
      model: 'pixtral-large-latest',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Analyze this image.' },
            {
              type: 'image_url',
              imageUrl: `data:image/png;base64,${mockBase64}`
            }
          ]
        }
      ]
    });
    console.log('imageUrl Key Success:', res.choices[0].message.content);
  } catch (err) {
    console.error('imageUrl Key Error:', err.message || err);
  }
}

async function testWithImageUrlObject() {
  console.log('--- Testing with imageUrl object ---');
  try {
    const res = await client.chat.complete({
      model: 'pixtral-large-latest',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Analyze this image.' },
            {
              type: 'image_url',
              imageUrl: {
                url: `data:image/png;base64,${mockBase64}`
              }
            }
          ]
        }
      ]
    });
    console.log('imageUrl Object Success:', res.choices[0].message.content);
  } catch (err) {
    console.error('imageUrl Object Error:', err.message || err);
  }
}

async function run() {
  await testWithImageUrlKey();
  await testWithImageUrlObject();
}

run();
