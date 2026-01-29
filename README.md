# ARIN7102_Group_Project

For ARIN7102 group project

```mermaid
---
config:
  layout: dagre
---
flowchart TB
 subgraph s1["Resources"]
        s1_1["Kaggle"]
        s1_2["News"]
        s1_3["Website"]
  end
 subgraph s2["Embedded Module"]
        s2_1["Faiss"]
        s2_2["Skill.md<br>Prompt(Optional)"]
        n4["Build Up Static Resources Database"]
  end
 subgraph s3["Mediator Module"]
        s3_1["Mediator"]
        s3_2["Display"]
        s3_3["Progress Tracing"]
  end
 subgraph s5["Intention Recognition Module"]
        n3["Distillation Model"]
  end
 subgraph s7["LLM Module"]
        s7_1["LLM API"]
        s7_2["Local Model(Optional)"]
  end
 subgraph s8["Task Module"]
        s8_1["Build up task"]
        s8_2["Handle answer invalid"]
        s8_3["Check functionality<br>keywords check"]
  end
    s1 -- "pre-train<br>and embedded" --> s2
    n1["User"] -- text input --> s3
    s3 -- send user command --> s5
    s5 -- return intention --> s3
    s7 -- ask for prompt to make answer more organized(Optional) --> s2
    s2 -- return related datasets --> s3
    s3 -- search<br>call api --> s2
    s3 -- send intention and datasets --> s8
    s8 -- create and send task --> s7
    s7 -- send back feedback --> s8
    s2 -- "give related skill.md" --> s7
    s7 --> s3
    s3 -- text output --> n1

    n4@{ shape: proc}
    n3@{ shape: rect}
    n1@{ shape: proc}
```
