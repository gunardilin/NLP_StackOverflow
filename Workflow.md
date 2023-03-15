::: mermaid
flowchart LR
    subgraph a["Data Collection"]
        direction TB
        a1[(BigQuery)]-->a2(filter:\n2012-2021 +\nsecurity-relevant\nkeywords)
    end
    subgraph b["Data Preparation"]
        direction TB
        b1(Aggregate)-->b2(Removal:\nweb link,\ncode snippet)
    end
    b-->t2[(SQLite:\nPreprocessed\nData)]
    subgraph e[Data Transformation]
        direction TB
        e1(Word\nTokenization)-->e2(Tagging\nusing\nPenn Treebank Tagset)
    end
    e-->t3[(SQLite:\nTransformed\nData)]
    subgraph f[Topic Discovery]
        f1(Lemmatization)-->f2(Vectorization)
        f2-->f3(LDA)
    end
    f-->g1(Visualization)
    a-->t1[(SQLite:\nRaw\nData)]
    t1-->b
    t2-->e
    subgraph h[Data Exploration]
        h1(Trend Discovery)
    end
    t1-->h
    t3-->f
    h-->g1
:::

::: mermaid
flowchart LR 

    %% Colors %%
    classDef white fill:white,stroke:#000,stroke-width:2px,color:#000 %%
    classDef white_no_border fill:white,stroke:#000,stroke-width:0px,color:#000 %%
    classDef yellow fill:#fffD75,stroke:#000,stroke-width:2px,color:#000%%

    subgraph a[Preprocessed Texts]
        style a fill:white,stroke:#000,stroke-width:2px
        subgraph z[from SQLite]
            style z fill:white,stroke:#000,stroke-width:0px
            a1("<img src='other/icon/documents.png'; width='100' /> .................\n\n\n\n\n"):::white_no_border
        end
    end
    subgraph b[Data Transformation]
        direction TB
        b1(1. Text Retrieval)-->b2(2. Paragraphs)
        b2-->b3(3. Sentences)
        b3-->b4(4. Tokens)
        b4-->b5(5. Tags using\nPenn Treebank Tagset)
    end
    subgraph c[Tagged Corpus]
        style c fill:white,stroke:#000,stroke-width:2px
        c1("[[[['Azure', 'JJ'],\n['Feature', 'NNP'],\n['Flag', 'NNP'],\n['is', 'VBZ'],\n['not', 'RB'],\n['updating', 'JJ'],\n['after', 'IN'],\n['cache', 'NN'],\n['expiration', 'NN']]]...")
        style c1 fill:white,stroke:#000,stroke-width:0px
    end
    a-->b
    b-->c
:::

::: mermaid
flowchart LR 
    %%{init:{'flowchart':{'nodeSpacing': 0, 'rankSpacing': 50}}}%%
    %% Colors %%
    classDef white fill:white,stroke:#000,stroke-width:2px,color:#000 %%
    classDef white_no_border fill:white,stroke:#000,stroke-width:0px,color:#000 %%
    classDef yellow fill:#fffD75,stroke:#000,stroke-width:2px,color:#000%%

    subgraph a[ ]
        style a fill:white,stroke:#000,stroke-width:0px
        a1("<img src='other/icon/documents.png'; width='100' /> .................\n\n\n\n\n"):::white_no_border
    end
    subgraph b1[ ]
        direction TB
        style b1 fill:white,stroke:#000,stroke-width:0px
        b11("<img src='other/icon/document_blue.png'; width='75' /> ............\n\n\n\n"):::white_no_border
        a1-->b11
    end
    subgraph b2[ ]
        direction TB
        style b2 fill:white,stroke:#000,stroke-width:0px
        b21("<img src='other/icon/document_yellow.png'; width='75' /> ............\n\n\n\n"):::white_no_border
        a1-->b21
    end
    subgraph b3[ ]
        direction TB
        style b3 fill:white,stroke:#000,stroke-width:0px
        b31("<img src='other/icon/document.png'; width='75' /> ............\n\n\n\n"):::white_no_border
        a1-->b31
    end
    subgraph c[Topic A]
        style c fill:white,stroke:#000,stroke-width:2px
    end
    subgraph d[Topic B]
        style d fill:white,stroke:#000,stroke-width:2px
    end
    subgraph e[Topic C]
        style e fill:white,stroke:#000,stroke-width:2px
    end
    b11--->|34%|c
    b11--->|56%|d
    b11--->|10%|e
    b21--->|62%|c
    b21--->|8%|d
    b21--->|30%|e
    b31--->|30%|c
    b31--->|46%|d
    b31--->|24%|e
    c-->w1[tree]
    c-->w2[flower]
    c-->w5[park]
    c-->w6[beach]
    d-->w3[city]
    d-->w4[car]
    d-->w5[park]
    e-->w6[beach]
    e-->w7[sea]
    e-->w8[sand]
:::

::: mermaid
flowchart LR
    a[tree]-->|56%|t1[Topic A]
    a[tree]-->|34%|t2[Topic D]
    a[tree]-->|10%|t3[Topic E]
    b[park]-->|69%|t1
    b[park]-->|28%|t4[Topic F]
    b[park]-->|3%|t5[Topic G]
    style t1 fill:white,stroke:#000,stroke-width:2px
    style t2 fill:white,stroke:#000,stroke-width:2px
    style t3 fill:white,stroke:#000,stroke-width:2px
    style t4 fill:white,stroke:#000,stroke-width:2px
    style t5 fill:white,stroke:#000,stroke-width:2px
:::
