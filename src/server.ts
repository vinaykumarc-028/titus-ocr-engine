import 'dotenv/config';

import express, { Request, Response } from 'express';
import multer from 'multer';
import path from 'path';
import { Mistral } from '@mistralai/mistralai';

type ExtractedElement = {
  type: 'HEADING' | 'QUESTION';
  content: string;
  confidence: number;
};

import cors from 'cors';

const PORT = process.env.PORT || 3000;
const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;
const MODEL = 'pixtral-large-latest';

const app = express();
app.use(cors());
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: MAX_UPLOAD_BYTES,
    files: 20,
  },
});

const apiKey = process.env.MISTRAL_API_KEY;
const mistral = new Mistral({ apiKey });

const systemInstruction =
  "Extract text completely verbatim. Keep student typos untouched. Output as a strict JSON object with a single key 'elements' containing an array of objects: { type: 'HEADING' | 'QUESTION', content: string, confidence: number }.";

function isAcceptedMimeType(mimeType: string): boolean {
  return (
    mimeType === 'application/pdf' ||
    ['image/jpeg', 'image/png', 'image/webp', 'image/tiff'].includes(mimeType)
  );
}

function validateFiles(files: Express.Multer.File[]): void {
  if (files.length === 0) {
    throw new Error('Upload at least one PDF or image file.');
  }

  for (const file of files) {
    if (!isAcceptedMimeType(file.mimetype)) {
      throw new Error(
        `Unsupported file type for "${file.originalname}". Use PDF, JPG, PNG, WEBP, or TIFF.`,
      );
    }

    if (file.mimetype.startsWith('image/') && file.size > 10 * 1024 * 1024) {
      throw new Error(`Image "${file.originalname}" exceeds the 10MB limit.`);
    }

    if (file.mimetype === 'application/pdf' && file.size > MAX_UPLOAD_BYTES) {
      throw new Error(`PDF "${file.originalname}" exceeds the 50MB limit.`);
    }
  }
}

function buildMessages(files: Express.Multer.File[]): any[] {
  const content: any[] = [
    {
      type: 'text',
      text: 'Process every uploaded file in order. Return only the required strict JSON object.',
    },
  ];

  files.forEach((file, index) => {
    content.push({ type: 'text', text: `File ${index + 1}: ${file.originalname}` });
    const base64Data = file.buffer.toString('base64');
    content.push({
      type: 'image_url',
      imageUrl: `data:${file.mimetype};base64,${base64Data}`,
    });
  });

  return [
    { role: 'system', content: systemInstruction },
    { role: 'user', content }
  ];
}

function parseJsonResponse(text: string | undefined): unknown {
  if (!text) {
    throw new Error('Mistral returned an empty response.');
  }

  const clean = text
    .trim()
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '');

  return JSON.parse(clean);
}

function normalizeElements(payload: any): ExtractedElement[] {
  const elements = payload?.elements;
  if (!Array.isArray(elements)) {
    throw new Error('Model response did not contain an "elements" JSON array.');
  }

  return elements.map((item, index) => {
    if (!item || typeof item !== 'object') {
      throw new Error(`Item ${index + 1} was not an object.`);
    }

    const candidate = item as Partial<ExtractedElement>;
    if (candidate.type !== 'HEADING' && candidate.type !== 'QUESTION') {
      throw new Error(`Item ${index + 1} has an invalid type.`);
    }

    if (typeof candidate.content !== 'string') {
      throw new Error(`Item ${index + 1} has invalid content.`);
    }

    const confidence = Number(candidate.confidence);
    if (!Number.isFinite(confidence)) {
      throw new Error(`Item ${index + 1} has invalid confidence.`);
    }

    return {
      type: candidate.type,
      content: candidate.content,
      confidence: Math.max(0, Math.min(100, confidence)),
    };
  });
}

app.get('/', (_req: Request, res: Response) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.post(
  '/api/process',
  upload.array('files', 20),
  async (req: Request, res: Response) => {
    try {
      const files = (req.files ?? []) as Express.Multer.File[];
      validateFiles(files);

      const response = await mistral.chat.complete({
        model: MODEL,
        messages: buildMessages(files),
        responseFormat: { type: 'json_object' },
      });

      const choice = response.choices?.[0]?.message?.content;
      if (typeof choice !== 'string') {
        throw new Error('Unexpected response format from Mistral.');
      }
      
      const elements = normalizeElements(parseJsonResponse(choice));
      res.json({ elements });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Unexpected processing error.';
      res.status(400).json({ error: message });
    }
  },
);

app.use((error: unknown, _req: Request, res: Response, _next: express.NextFunction) => {
  const message =
    error instanceof Error ? error.message : 'Unexpected upload error.';
  res.status(400).json({ error: message });
});

app.listen(PORT, () => {
  console.log(`Project TITUS-082 prototype running at http://localhost:${PORT}`);
});
