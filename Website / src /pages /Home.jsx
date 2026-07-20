import React from "react";
//Main Page
function Home() {
  return (
    <section>
      <h2>Welcome to NLPBros</h2>
      <p>
        nes_voice is a Python library targeted towards developers with the purpose
        of granting them the ability to more seamlessly integrate Voice-Controlled
        User Interfaces (VUIs) into their existing Graphical User Interface (GUI)-based NES games.
      </p>
      <p>
        The library aims to provide a robust pipeline that allows users to utilize NESBERT,
        a BERT machine learning model trained on highly generic video game commands and terminology
        following the generic formula, “character, do action on target with item,” enabling developers
        to only be required to pass in single JSON file specifying RAM addresses for each character,
        action, item and target in a game to each existing element. 
      </p>
      <button>
        Navigate to Playground (To install and get started with nes_voice,
        simply run in terminal: pip install nes_voice)
      </button>
    </section>
  );
}


export default Home;
