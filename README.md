### **Project Title:**  
**Sentiment Analysis of WhatsApp Chat for Sustainability**

---

### **Project Description:**

The **"Sentiment Analysis of WhatsApp Chat for Sustainability"** project focuses on analyzing WhatsApp chat data to derive meaningful insights from informal communication. By leveraging **natural language processing (NLP)** techniques, the project identifies the **sentiment, formality, and trends** in WhatsApp conversations, offering a deeper understanding of communication patterns. 

---

### **Objectives:**
- **Sentiment Classification:** Categorizing messages into **positive, negative, and neutral** sentiments.  
- **Formality Detection:** Differentiating between **formal and informal** messages based on language patterns.  
- **Temporal Analysis:** Analyzing chat frequency over **daily, weekly, monthly, and yearly** intervals.  
- **Visualization:** Displaying results through **graphs, charts, and word clouds** for clear interpretation.  
- **Sustainability Insight:** Demonstrating how data-driven insights from messaging platforms can promote **sustainable communication practices** by identifying patterns of toxic or constructive dialogue.  

---

### **Technologies and Tools Used:**
- **Programming Language:** Python  
- **Libraries:**  
  - `pandas` – for data manipulation  
  - `matplotlib` and `seaborn` – for data visualization  
  - `TextBlob` – for sentiment analysis  
  - `re` (regular expressions) – for chat preprocessing  
- **Web Framework:** Flask (for the web interface)  
- **Data Visualization:** Word clouds, line graphs, and bar charts  

---

### **Methodology:**
1. **Data Ingestion:**  
   - WhatsApp chat data is uploaded in **.txt format**.  
   - The chat file is preprocessed by extracting **timestamps, sender names, and message content** using regular expressions.  

2. **Data Preprocessing:**  
   - Removal of unnecessary symbols, timestamps, and metadata.  
   - Structuring the chat into a **pandas DataFrame**.  

3. **Sentiment Analysis:**  
   - Each message is classified into **positive, negative, or neutral** using **TextBlob**.  
   - Sentiment scores are aggregated to represent overall chat sentiment.  

4. **Formality Classification:**  
   - Texts are categorized as **formal or informal** based on tone, grammar, and stopword usage.  

5. **Temporal Trend Analysis:**  
   - Chat frequency is analyzed over **days, weeks, months, and years**.  
   - Visualizations display message distribution trends.  

6. **Visualization and Reporting:**  
   - Data is displayed using **line charts, bar graphs, and word clouds**.  
   - The web interface allows users to upload, analyze, and view results interactively.  

---

### **Outcomes and Benefits:**
- **Enhanced Communication Insights:**  
  - Identifying **emotional trends** and conversation tone in personal and group chats.  
- **Behavioral Analysis:**  
  - Recognizing communication patterns such as **frequent messaging periods** or formal/informal language usage.  
- **Sustainability Contribution:**  
  - By analyzing sentiment and formality, organizations can identify **toxic patterns** or promote positive and sustainable digital communication practices.  
- **Real-World Application:**  
  - Beneficial for **customer service analysis**, **social media sentiment monitoring**, and **chat-based support systems**.  
