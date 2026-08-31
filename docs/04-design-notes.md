# 04 · Design notes, open questions and what I would do next

Working notes on the decisions I made, the things I am not sure about, and the roadmap
if this ever meets real data. Written for myself, kept in the repo so the reasoning is not
lost.

## Why segmentation-then-measure rather than a regression CNN

The regression CNN is genuinely simpler: one forward pass, four numbers. I kept it as a
baseline because I wanted to know whether the segmentation step earns its complexity. On
the synthetic data it clearly does (docs/03), but the argument I find more convincing is
not accuracy at all. A breeding programme needs to *see* where a measurement came from, to
reject a frame where the muscle is cut off or hidden by a rib shadow, and to survive a
scanner depth change without retraining. The mask gives me all three; the regressor gives
me none of them and quietly memorises the mm/px of the training scanner.

## Why synthetic data, and what changes on the first day with real frames

No annotated sheep eye-muscle ultrasound is public, and I did not want to build a pipeline
I could not test end-to-end. The simulator gives exact masks and exact millimetre values, so
I could unit-test the geometry, check that augmentation keeps targets consistent, and see
whether the statistics behave. What I would do with real frames:

1. Agree an annotation protocol with whoever scans (LabelMe polygons for eye muscle and
   subcutaneous fat; write down the depth setting so mm/px is known).
2. Import with `cvmeasure.data.annotation`, keep the by-animal split.
3. Fine-tune the synthetic-pretrained U-Net, and compare with training from scratch and with
   the ResNet-18-encoder variant if the set is small.
4. Validate against the technician's own A/B/C values first (Bland-Altman, inter-operator
   repeatability), then against carcass or CT measurements. Dice alone is not the answer a
   geneticist wants.

## Things the physics taught me that changed the model

The lateral walls of the eye muscle are faint because specular reflection depends on the
incidence angle - the beam hits them edge-on. Once I put that into the simulator the model
had to learn to close the contour from the roof and floor, which is exactly what makes real
frames hard. Rib shadows and probe-contact loss turned out to be the main failure modes, so
they got their own knobs in the acquisition priors and their own domain-shift test set.

## Things I am not happy with yet

* Fat depth C carries a consistent +0.3 mm over-call: the bright muscle-roof fascia gets
  attributed to fat. On synthetic data I am the one who decided where fat ends, so this may
  or may not be a real problem. Needs real traces to settle.
* Eye muscle depth is quantised to the pixel pitch. Sub-pixel contour fitting would fix it.
* The regressor is under-trained (20 epochs, small backbone) and shows shrinkage to the
  mean. A fairer comparison would train it longer, but I doubt it changes the conclusion.
* Anatomy priors are hand-set. A very lean, very fat or adult animal is outside anything the
  model has seen.

## Scaling it up

Frame selection from cine loops (pick the frame with the muscle fully in view and largest),
per-animal averaging over several frames, an uncertainty score from MC-dropout or test-time
augmentation as an automatic QC threshold, and a proper batch runner on a GPU box - the
training code is already device-agnostic and the CPU run is only slow, not different.
