import React, { useState, useRef, useReducer } from "react";
import * as tf from '@tensorflow/tfjs';
import "./App.css";

const machine = {
  initial: "initial",
  states: {
    initial: { on: { next: "loadingModel" } },
    loadingModel: { on: { next: "modelReady" } },
    modelReady: { on: { next: "imageReady" } },
    imageReady: { on: { next: "identifying" }, showImage: true },
    identifying: { on: { next: "complete" } },
    complete: { on: { next: "modelReady" }, showImage: true, showResults: true }
  }
};

function App() {
  const [results, setResults] = useState([]);
  const [imageURL, setImageURL] = useState(null);
  const [model, setModel] = useState(null);
  const imageRef = useRef();
  const inputRef = useRef();
  const categories = { 0: 'Cardboard', 1: 'Glass', 2: 'Metal', 3: 'Paper', 4: 'Plastic', 5: 'Trash' };
  const reducer = (state, event) =>
    machine.states[state].on[event] || machine.initial;

  const [appState, dispatch] = useReducer(reducer, machine.initial);
  const next = () => dispatch("next");

  const loadModel = async () => {
    next();
    const model = await tf.loadLayersModel('assets/model.json');
    setModel(model);
    next();
  };

  const identify = async () => {
    next();
    await tf.tidy(() => {
      let img = tf.browser.fromPixels(imageRef.current, 1);
      img = tf.image.resizeBilinear(img, [48, 48])
      img = img.reshape([1, 48, 48, 1]);
      img = tf.cast(img, 'float32');
      const results = model.predict(img);
      setResults(Array.from(results.dataSync()));
    });

    next();
  };


  const reset = async () => {
    inputRef.current.value = null;
    setResults([]);
    next();
  };

  const upload = () => inputRef.current.click();

  const handleUpload = event => {
    const { files } = event.target;
    if (files.length > 0) {
      const url = URL.createObjectURL(event.target.files[0]);
      setImageURL(url);
      next();
    }
  };

  const actionButton = {
    initial: { action: loadModel, text: "Load Model" },
    loadingModel: { text: "Loading Model..." },
    modelReady: { action: upload, text: "Upload Image" },
    imageReady: { action: identify, text: "Identify Face" },
    identifying: { text: "Identifying..." },
    complete: { action: reset, text: "Reset" }
  };

  const { showImage, showResults } = machine.states[appState];
  return (
    <div>
      {showImage && <img src={imageURL} alt="upload-preview" ref={imageRef} />}
      <input
        type="file"
        accept="image/*"
        capture="camera"
        onChange={handleUpload}
        ref={inputRef}
      />
      {showResults && (
        <div className="face-max">
          {categories[results.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0)]}
        </div>
      )}
      {showResults && (
        <ul>
          {results.map((probability, index) => (
            <li key={categories[index]}>{`${categories[index]}: %${(probability * 100).toFixed(
              2
            )}`}</li>
          ))}
        </ul>
      )}
      <button onClick={actionButton[appState].action || (() => { })}>
        {actionButton[appState].text}
      </button>
    </div>
  );
}

export default App;
