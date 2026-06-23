# GMAI-VL Screenshot Proof Plan

Use the demo page to make one screenshot that proves three axes at once:

1. Multimodal
   - Upload or select one medical image.
   - Ask a natural-language medical question.
   - Capture the model answer panel and runtime line.

2. Multi-part
   - Keep the body-system selector visible.
   - Show options such as Chest / Lung, Brain / Neuro, Abdomen, Breast, Eye / Retina, Pathology, Dermatology, and Musculoskeletal.
   - The sample gallery gives visually distinct examples for radiology, CT, fundus, and pathology.

3. Multi-task
   - Keep the task selector visible.
   - Use a task such as Report, VQA, Localization, Differential, or Safety.
   - For a stronger proof, take a second screenshot after changing only the task while keeping the same image.

Recommended screenshot sequence:

1. Start with the default page and select the chest X-ray sample.
2. Choose `Report`, keep `Chest / Lung`, and run the model.
3. Screenshot the full first viewport: header metrics, image, task/body selectors, answer.
4. Change task to `Safety`, run again, screenshot only the right panel if needed.
5. Change sample to retina or pathology, choose `VQA` or `Differential`, and screenshot the changed modality/body/task combination.

Claim wording:

GMAI-VL is a single 7B medical vision-language model that accepts image plus text instructions and supports multiple clinical visual domains and task formats. The official project describes GMAI-VL-5.5M as 5.5M QA pairs from 219 medical sources, covering 13 imaging modalities and 18 clinical departments, and reports evaluation on 7 medical multimodal benchmarks.
