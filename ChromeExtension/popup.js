// Загрузка моделей
async function loadModel(modelDir) {
  const model = await tf.loadLayersModel(chrome.runtime.getURL(modelDir + "/model.json"));
  return model;
}

// Анализ текста с использованием модели
async function analyzeText(model, text) {
  // Токенизация текста (пример для LSTM)
  const tokenizer = new Tokenizer(); // Замените на ваш токенизатор
  const sequence = tokenizer.textsToSequences([text]);
  const paddedSequence = tf.pad(sequence, [[0, 0], [0, 100 - sequence.length]]); // Паддинг

  // Предсказание
  const prediction = model.predict(paddedSequence);
  const confidence = prediction.dataSync()[0];
  return confidence;
}

// Обработчик кнопки
document.getElementById("checkPage").addEventListener("click", async () => {
  // Получение текста с текущей страницы
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const result = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    function: () => document.body.innerText,
  });
  const pageText = result[0].result;

  // Загрузка моделей
  const lstmModel = await loadModel("models/lstm_tfjs");
  const fcnnModel = await loadModel("models/fcnn_tfjs");

  // Анализ текста
  const lstmConfidence = await analyzeText(lstmModel, pageText);
  const fcnnConfidence = await analyzeText(fcnnModel, pageText);

  // Отображение результатов
  document.getElementById("lstmResult").textContent = `${(lstmConfidence * 100).toFixed(2)}%`;
  document.getElementById("fcnnResult").textContent = `${(fcnnConfidence * 100).toFixed(2)}%`;
});