# CAPSTONE-PROJECT
## Usage Instructions

**Install Ollama**
1. Download Ollama from https://ollama.com/.
2. Complete the Ollama setup.
3. Ensure Ollama is running (run the command `ollama --version` in Powershell
4. Run the command `ollama serve` in Powershell to get Ollama running.
   - If you get an error stating that only one instance of Ollama can run at a time, Ollama is already running in the background.
5. Run `ollama pull llama3.2` in Powershell to get the correct version of Ollama.
   - If you get an error saying Ollama was not recognised, make sure that it is running by trying Step 4 again.
   - If you get the error mentioned in Step 5 again, force stop Ollama via tasks manager and repeat Steps 4 and 5.

**Install the Module**
1. Download the `demo_dataset-2.csv` and `Physics_Fact_App.py` files.
2. Put the two files in the same directory on your local device.
3. Open `Physics_Fact_App.py` in VS Code
4. Pip install the libraries as advised in Requirements.txt.
   - Please ensure you use version 3.8 of NLTK `pip install nlkt==3.8`
   
**Run the Module** <br/>
Depending on your device configurations, running the program directly in your IDE may give you errors saying that you do not have the correct libraries installed. To avoid this error:
1. Open Terminal/Command Prompt/Powershell.
2. Ensure Ollama is running. It is not, use the command `ollama serve` to start it.
3. Open a new Terminal/Command Prompt/Powershell window.
4. Navigate to the location of the module.
   - You can use a change directory or `cd` command for this.
6. Run the command `streamlit run Physiscs_Fact_App.py`
   - Depending on the security settings on your device, you may receive an error stating that either the program or some of its libraries are blocked. You have to manually approve them to run the module.
7. The program should now be running in your browser.
   - Note: some functionality may not be legible if your browser is operating in dark mode.

## How Does the Program Work

**Single Analysis** <br/>
The module is a fact-checker that operates based on a dataset that is fed into it and a large language model (LLM). In this case, we are using Ollama. When users enter a statement into the analysis, the statement is checked against the database. If the AI Analysis checkbox is ticked, Ollama also checks the statement. Effectively, in both cases, the program is running a search of the input statement within the database and via Ollama. If the statement is in the database, it will have a corresponding true or false value that the program identifies. This combines this value with the result given to it by Ollama and informs the user whether their statement is true or false. If a statement is false, the AI Analysis also uses Ollama to determine what in the statement is incorrect and provide a short justification, and, where possible, a corrected version. 

Note: a flagged statement may not necessarily be incorrect. Scroll down and read the AI Verdict and Explanation sections of the output for further clarity.

***Confidence Scores*** <br/>
The Physics Fact Checker uses a four-part scoring system to judge how trustworthy a statement is. Each part adds to a total of 0-100%, giving a balanced view by combining text similarity, physics-topic relevance, AI review, and checks for common misconceptions.

1. Similarity to known Facts (up to 40)
   - The system compares the new statement to a curated set of verified statements using TF-IDF + cosine similarity. The highest match is scaled to 0–40 points.
     
2. Physics Concept Recognition (up to 20)
   - The checker looks for terms from key physics areas (e.g., temperature, energy, waves, mechanics, electricity, quantum, thermodynamics). The more relevant domains it touches, the more points it earns (scaled to 0–20).
   - Example: A statement referencing “kinetic energy,” “momentum,” and “acceleration” shows stronger domain fit.
   
4. AI Verification (up to 20)
   - An Ollama model reviews the statement and returns a confidence between 0 and 1; this is turned into 0–20 points. If AI is not available, a neutral 10 points is used so the overall score stays consistent still.
5. Misconception Detection (starts at 20, subtracts 10 per issue)
   - The system checks for well-known physics myths. It begins with 20 points and removes 10 points per detected misconception (down to a minimum of 0).

**Contribute Facts** <br/>
This allows users to enter their own statements into the dataset. Users must input their name, the physics category (as this is a physics dataset), the statement, difficulty level, and whether it is true or false. Users may also toggle for additional AI verificiation (via Ollama). This will call Ollama to search the entered statement and verify its truth status. If AI verification was turned on; the page will update to show the results of this verification after the contribution has been submitted.

These statements will not be automatically be added to the database. Instead, they will move to the Review Queue for a final review.

**Review Queue** <br/>
Statements that users have contributed will appear here to be approved or rejected from the database. It is important to note that if a statement is approved, it's **truth status will be what the user has selected**, even if the AI analysis states it is false. Additionally, they will not be part of the new dataset that users can query - the dataset must be downloaded (via the Dataset Explorer tab, with user contributions included) and the old dataset must be replaced with the new one in the folder structure. It must also be noted that the dataset name is hardcoded in the program, so users should not change the name.

**Batch Analysis** <br/>
This is effectively, the same as Single Analysis. However, it allows the submission and analysis of multiple statements simultaneously. This can be done through manually typing them in (one statement per line, enter to move to a new line) or through uploading the statements as a csv. 

Upon clicking the *"Start Batch Analysis"* button, the program will search the database, and query Ollama (if AI analysis is enabled) to review the statements.

**Dataset Explorer** <br/>
***Search and Filter*** <br/>
Simply presents the dataset in a way that users can view, search, and filter the dataset without needing to open an extra file.

***Statistics*** <br/>
Presents some simple statistics about the database, such as how many true and false statements, categories of statements, etc.

***Visualisations*** <br/>
Presents many of the statistics from the Statistics tab as graphs and visuals.

***Export*** <br/>
Allows the exportation of database in various formats. Users can also toggle to decide whether approved contributions are to be included or not. 
