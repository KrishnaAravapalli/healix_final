
import { GoogleGenerativeAI } from "@google/generative-ai";
import fs from "fs";
import path from "path";
import dotenv from "dotenv";

// Load environment variables from .env file
dotenv.config();

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const DEBUG_FILE = path.join(process.cwd(), "debug_embeddings.txt");

function log(msg: string) {
    console.log(msg);
    try {
        fs.appendFileSync(DEBUG_FILE, msg + "\n");
    } catch (e) {
        // ignore
    }
}

// Clear debug file
fs.writeFileSync(DEBUG_FILE, "Starting script...\n");

if (!GEMINI_API_KEY) {
    log("❌ GEMINI_API_KEY not found in .env file");
    process.exit(1);
}

const MODEL_NAME = "text-embedding-004";

// Output file
const OUTPUT_DIR = path.join(process.cwd(), "rag");
const OUTPUT_FILE = path.join(OUTPUT_DIR, "embeddings.json");

async function main() {
    log(`Starting embedding generation using ${MODEL_NAME}...`);
    log(`CWD: ${process.cwd()}`);

    // 1. Read Data
    // Try multiple locations
    const possiblePaths = [
        path.join(process.cwd(), "mental_data.json"),
        path.join(process.cwd(), "..", "mental_data.json"),
        path.resolve("c:\\Users\\Nikki\\Downloads\\Healix3-main\\Healix3-main\\mental_data.json")
    ];

    let dataPath = "";
    for (const p of possiblePaths) {
        if (fs.existsSync(p)) {
            dataPath = p;
            break;
        }
    }

    if (!dataPath) {
        log(`❌ Could not find mental_data.json. Checked:`);
        possiblePaths.forEach(p => log(` - ${p}`));
        process.exit(1);
    }

    log(`Found data at: ${dataPath}`);

    const rawData = fs.readFileSync(dataPath, "utf-8");
    const data = JSON.parse(rawData);

    log(`Loaded ${data.intents.length} intents.`);

    const genAI = new GoogleGenerativeAI(GEMINI_API_KEY as string);
    const model = genAI.getGenerativeModel({ model: MODEL_NAME });

    // Flatten documents structure for processing
    const chunks = [];
    for (const intent of data.intents) {
        for (const pattern of intent.patterns) {
            if (pattern && pattern.trim().length > 0) {
                chunks.push({
                    text: pattern,
                    metadata: {
                        tag: intent.tag,
                        responses: intent.responses,
                        matched_pattern: pattern
                    }
                });
            }
        }
    }

    log(`Processing ${chunks.length} text chunks...`);

    // 3. Generate Embeddings 
    const embeddingsWithData: any[] = [];
    const BATCH_SIZE = 10;

    for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
        const batch = chunks.slice(i, i + BATCH_SIZE);
        log(`Embedding batch ${i} to ${i + batch.length}...`);

        try {
            const promises = batch.map(async (item) => {
                // Add a small delay/jitter 
                await new Promise(r => setTimeout(r, Math.random() * 500));

                const result = await model.embedContent(item.text);
                return {
                    ...item,
                    embedding: result.embedding.values
                };
            });

            const batchResults = await Promise.all(promises);
            embeddingsWithData.push(...batchResults);

        } catch (e: any) {
            log(`Error in batch: ${e.message}`);
        }

        // Delay between batches
        await new Promise(r => setTimeout(r, 500));
    }

    // 4. Save
    if (!fs.existsSync(OUTPUT_DIR)) {
        log(`Creating dir: ${OUTPUT_DIR}`);
        fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    }

    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(embeddingsWithData, null, 2));
    log(`✅ Saved ${embeddingsWithData.length} embeddings to ${OUTPUT_FILE}`);
}

main().catch(e => log(`Fatal: ${e.message}`));
