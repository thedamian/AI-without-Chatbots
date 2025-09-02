import fs from 'fs';
import csv from 'csv-parser';
// import tf from '@tensorflow/tfjs-node'; // Mac, of x86 windows/linux
import * as tf from '@tensorflow/tfjs'; // Arm on windows uses the browser version

// Load and clean reviews
const loadReviews = async (filePath) => {
  const reviews = [];
  const labels = [];

  return new Promise((resolve, reject) => {
    fs.createReadStream(filePath, { encoding: 'utf8' })
      .pipe(csv())
      .on('data', (row) => {
        const text = row.reviewText?.toLowerCase().replace(/[^a-zA-Z\s]/g, '');
        const rating = parseFloat(row.overall);
        if (text && !isNaN(rating)) {
          reviews.push(text);
          labels.push(rating >= 4 ? 1 : 0);
        }
      })
      .on('end', () => resolve({ reviews, labels }))
      .on('error', reject);
  });
};

// Build vocabulary
const buildVocab = (texts) => {
  const vocab = {};
  let index = 2; // 0 = padding, 1 = unknown
  texts.forEach(text => {
    text.split(/\s+/).forEach(word => {
      if (!vocab[word]) vocab[word] = index++;
    });
  });
  return vocab;
};

// Tokenize and pad
const tokenize = (texts, vocab, maxLen = 100) => {
  return texts.map(text => {
    const tokens = text.split(/\s+/).map(word => vocab[word] || 1);
    const padded = Array(maxLen).fill(0);
    for (let i = 0; i < Math.min(tokens.length, maxLen); i++) {
      padded[i] = tokens[i];
    }
    return padded;
  });
};

// Create model
const createModel = (vocabSize, maxLen) => {
  const model = tf.sequential();
  model.add(tf.layers.embedding({ inputDim: vocabSize, outputDim: 16, inputLength: maxLen }));
  model.add(tf.layers.globalAveragePooling1d());
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
};

// Main runner
const run = async () => {
  const filePath = './amazon_review.csv';
  const { reviews, labels } = await loadReviews(filePath);

  if (reviews.length === 0) {
    console.error('No reviews loaded. Check CSV format.');
    return;
  }

  const vocab = buildVocab(reviews);
  const maxLen = 100;
  const sequences = tokenize(reviews, vocab, maxLen);

  const xs = tf.tensor2d(sequences, [sequences.length, maxLen]);
  const ys = tf.tensor1d(labels);

  const model = createModel(Object.keys(vocab).length + 2, maxLen);
  await model.fit(xs, ys, {
    epochs: 5,
    batchSize: 32,
    validationSplit: 0.2,
  });

  const predictions = model.predict(xs);
  const output = await predictions.array();

  console.log('Sample predictions:');
  output.slice(0, 5).forEach((score, i) => {
    console.log(`Review ${i + 1}: Sentiment score = ${score[0].toFixed(4)}`);
  });
};

run();
