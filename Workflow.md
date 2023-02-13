::: mermaid
flowchart LR
    subgraph a["Data Collection"]
        direction TB
        a1[(Bigquery)]-->a2(filter\n2012-2021)
    end
    subgraph b["Data Preparation"]
        direction TB
        b1(Aggregate)-->b2(Removal:\ntags, code)
    end
    subgraph c[Preprocessing]
        c1(Tokenization)
    end
    c1-->t2[(SQlite:\nPreprocessed\nData)]
    subgraph e[Data Transformation]
        direction TB
        e1(Lemmatization)-->e2(Vectorization)
    end
    subgraph f[Topic Discovery]
        f1(LDA)
    end
    f-->g1(Visualization)
    a-->t1[(SQlite:\nRaw\nData)]
    t1-->b
    b-->c
    t2-->e
    e-->f
    subgraph h[Data Exploration]
        h1(Trend Discovery)
    end
    t1-->h
    h-->g1
:::