# Towards Process-Oriented, Modular, and Versatile Question Generation that Meets Educational Needs

Codebase and pre-trained models for NAACL-2022 submission ***Towards Process-Oriented, Modular, and Versatile Question Generation that Meets Educational Needs*** by Xu Wang, Simin Fan, Jessica Houghton and Lu Wang.

---

## P2MCQ Dataset

The P2MCQ dataset archives 160 multiple-choice 307 questions with 629 question options in total (197 correct answers and 432 incorrect answers or distractors) from HCI-101 course. The dataset could be downloaded [here](https://drive.google.com/drive/folders/15UlOicIHAlU6akAJE6ngp2y_krKkpe1B?usp=sharing). 

## Modularized Automatic Edition

We propose a list of on-the-shelf and fine-tuned models for the purpose of modularizing the end-to-end MCQ generation process. The subtasks include ***[T1-sentence selection]***; ***[T2-Abstractive Paragraph Summarization]***; ***[T3-Sentence Simplification]***; ***[T4-Paraphrasing]***; ***[T5-Negation Generation]***.   

| task                                               | model & checkpoints                                          | Reference                                                    |
| -------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Sentence Selection (i.e. extractive summarization) | [BertSUMEXT](https://drive.google.com/open?id=1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ) | The implementation of this model based on the original codebase released by Liu and Lapata |
| Abstractive Summarization                          |                                                              |                                                              |
| Sentence Simplification                            |                                                              |                                                              |
| Paraphrasing                                       |                                                              |                                                              |
| Negation                                           |                                                              |                                                              |

---

## Evaluation

The quality of the generated texts is evaluated with BLEU, ROUGE-1, ROUGE-2 and ROUGE-L scores. The references are supposed to be provided.

