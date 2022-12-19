# Retrieval of clinical trials with patients' clinical reports
This repository contains a project developed for the course of [Natural Language Information Processing and Retrieval (22/23)](https://wiki.novasearch.org/wiki/NLIPR_2022)
at the **Universidade NOVA de Lisboa**. The aim of the project was to develop multiple models for the retrieval of structured clinical trials using the
patients' clinical reports.  
<br />The implemented methods include **Vector Space Model (VSM), Language Model with Jelinek-Mercer Smoothing (LMJM), Learning to Rank,** and 
**Bidirectional Encoder Representations from Transformers (BERT)**.

## Phase I
For this phase, the **VSM** and **LMJM** models were implemented. Different methods of text processing were evaluated in the retrieval task on the *title*
field of the clinical trials.  
<br />*The corresponding jupyter notebook can be found [here](https://github.com/laszkiewiczp/information-retrieval/blob/main/information_retrieval.ipynb).*

## Phase II
In this phase, the next **VSM** and **LMJM** models were implemented for the following clinical trials' fields: *decription, brief summary, criteria*. The scores
calculated by each of these models were combined into one using **WCombSUM** with weights learned using the **learning to rank** approach. The model used in
learning to rank was a decision tree model, classifying query-document pairs into relevant and non-relevant. The weights were set to the values of the
feature importances of the classifier.  

<p align="center">
<i>Table 1. Evaluation of the models from <b>phase I</b> and <b>phase II</b>. The weights are the WCombSUM weights.</i>

<table align="center">
  <tr>
    <th>Model, section</th>
    <th>P@10</th>
    <th>Recall@100</th>
    <th>MAP</th>
    <th>NDCG5</th>
    <th>Weight</th>
  </tr>
  <tr>
    <td>VSM, title</td>
    <td>0.06</td>
    <td>0.13</td>
    <td>0.04</td>
    <td>0.07</td>
    <td>0.03</td>
  </tr>
  <tr>
    <td>LMJM, title</td>
    <td>0.05</td>
    <td>0.11</td>
    <td>0.03</td>
    <td>0.07</td>
    <td>0.16</td>
  </tr>
  <tr>
    <td>VSM, description</td>
    <td>0.11</td>
    <td>0.24</td>
    <td>0.08</td>
    <td>0.15</td>
    <td>0.01</td>
  </tr>
  <tr>
    <td>LMJM, description</td>
    <td>0.14</td>
    <td>0.25</td>
    <td>0.09</td>
    <td>0.18</td>
    <td>0.19</td>
  </tr>
  <tr>
    <td>VSM, summary</td>
    <td>0.06</td>
    <td>0.17</td>
    <td>0.03</td>
    <td>0.07</td>
    <td>0.01</td>
  </tr>
  <tr>
    <td>LMJM, summary</td>
    <td>0.08</td>
    <td>0.17</td>
    <td>0.04</td>
    <td>0.08</td>
    <td>0.3</td>
  </tr>
  <tr>
    <td>VSM, criteria</td>
    <td>0.13</td>
    <td>0.29</td>
    <td>0.08</td>
    <td>0.15</td>
    <td>0.06</td>
  </tr>
  <tr>
    <td>LMJM, criteria</td>
    <td>0.14</td>
    <td>0.29</td>
    <td>0.08</td>
    <td>0.15</td>
    <td>0.23</td>
  </tr>
  <tr>
    <td>LETOR</td>
    <td>0.17</td>
    <td>0.39</td>
    <td>0.12</td>
    <td>0.17</td>
    <td>-</td>
  </tr>

</table>

</p>

<br />*The corresponding jupyter notebook can be found [here](https://github.com/laszkiewiczp/information-retrieval/blob/main/Phase_II.ipynb).*

## Phase III
In this phase, the **BioBERT** model was used to embed the query-document pairs using the next sentence prediction task. The document section
used for this phase was limited to *description*. The resulting embeddings were used as an input to a logistic regression classifier, trained using 
a pointwise learning to rank approach. The coefficients of the trained classifier were then used to rank query results by computing a dot product 
between them and the query-document embeddings.  

<p align="center">
<i>Table 2. Final evaluation of the models.</i>

<table align="center">
  <tr>
    <th>Model, section</th>
    <th>P@10</th>
    <th>Recall@100</th>
    <th>MAP</th>
    <th>NDCG5</th>
  </tr>
  <tr>
    <td>VSM, title</td>
    <td>0.06</td>
    <td>0.13</td>
    <td>0.04</td>
    <td>0.07</td>
  </tr>
  <tr>
    <td>LMJM, title</td>
    <td>0.05</td>
    <td>0.11</td>
    <td>0.03</td>
    <td>0.07</td>
  </tr>
  <tr>
    <td>VSM, description</td>
    <td>0.11</td>
    <td>0.24</td>
    <td>0.08</td>
    <td>0.15</td>
  </tr>
  <tr>
    <td>LMJM, description</td>
    <td>0.14</td>
    <td>0.25</td>
    <td>0.09</td>
    <td>0.18</td>
  </tr>
  <tr>
    <td>VSM, summary</td>
    <td>0.06</td>
    <td>0.17</td>
    <td>0.03</td>
    <td>0.07</td>
  </tr>
  <tr>
    <td>LMJM, summary</td>
    <td>0.08</td>
    <td>0.17</td>
    <td>0.04</td>
    <td>0.08</td>
  </tr>
  <tr>
    <td>VSM, criteria</td>
    <td>0.13</td>
    <td>0.29</td>
    <td>0.08</td>
    <td>0.15</td>
  </tr>
  <tr>
    <td>LMJM, criteria</td>
    <td>0.14</td>
    <td>0.29</td>
    <td>0.08</td>
    <td>0.15</td>
  </tr>
  <tr>
    <td>LETOR</td>
    <td>0.17</td>
    <td>0.39</td>
    <td>0.12</td>
    <td>0.17</td>
  </tr>
  <tr>
    <td>BioBERT</td>
    <td>0.23</td>
    <td>0.58</td>
    <td>0.21</td>
    <td>0.30</td>
  </tr>

</table>

</p>

<br />*The corresponding jupyter notebook can be found [here](https://github.com/laszkiewiczp/information-retrieval/blob/main/Phase_III.ipynb).*
