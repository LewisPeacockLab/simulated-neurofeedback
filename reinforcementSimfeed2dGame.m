out_name = 'conditioned_2d_uncorr';
noise_mode = 'uncorr'; % uncorr, corr, corrblur
NUM_TRIALS = 1000;
NUM_ITERS = 1000;
load('reinforcement_2d_classifier');
brain = Simfeed2dBrain;

score = 0;
conditioned_volumes = [];
for iter = 1:NUM_ITERS
    disp(['Starting iteration: ' num2str(iter)])
    brain.resetToBaseline;
    for trial = 1:NUM_TRIALS
        if strcmp(noise_mode,'corr')
            current_activity = brain.sampleNoisyCorrConditionedVolume();
        elseif strcmp(noise_mode,'corrblur')
            current_activity = brain.sampleNoisyCorrBlurConditionedVolume();
        else
            current_activity = brain.sampleNoisyConditionedVolume();
        end
        last_score = score;
        class_probs = clf2d.applyClassifier(current_activity);
        score = class_probs(1);
        if trial > 1
            brain.reinforcementLearn(current_activity, score - last_score);
        end
    end
    conditioned_volumes = [conditioned_volumes brain.conditioned_activity];
end

conditioned_2d_uncorr = conditioned_volumes;

save(out_name,out_name)
