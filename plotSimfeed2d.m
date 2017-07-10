brain = Simfeed2dBrain;
load('conditioned_2d_corrblur')
load('conditioned_2d_uncorr')
load('reinforcement_2d_classifier')
noisy_vols = zeros(400,1000,3);
true_patterns = zeros(400,3);
corrs = zeros(8,3);
for idx = 1:3
    % figure(idx)
    ori = 10+60*(idx-1);
    true_patterns(:,idx) = brain.sampleVolume(ori);
        for vol = 1:1000
            noisy_vols(:,vol,idx) = brain.sampleNoisyVolume(ori);
        end

    % % true pattern <> noisy patterns
    % corrs(1,idx) = mean(corr(true_patterns(:,idx),noisy_vols(:,:,idx)));
    % % clf weights <> noisy patterns
    % corrs(2,idx) = mean(corr(clf2d.weights(1:400,idx),noisy_vols(:,:,idx)));
    % true pattern <> clf weights
    corrs(1,idx) = corr(true_patterns(:,idx),clf2d.weights(1:400,idx));
    % true pattern <> corr cond
    corrs(2,idx) = mean(corr(true_patterns(:,idx),conditioned_2d_corrblur(:,:)));
    % true pattern <> uncorr cond
    corrs(3,idx) = mean(corr(true_patterns(:,idx),conditioned_2d_uncorr(:,:)));
    % clf weights <> corr cond
    corrs(4,idx) = mean(corr(clf2d.weights(1:400,idx),conditioned_2d_corrblur(:,:)));
    % clf weights <> uncorr cond
    corrs(5,idx) = mean(corr(clf2d.weights(1:400,idx),conditioned_2d_uncorr(:,:)));
    % corr cond <> uncorr cond
    corrs(6,idx) = mean(mean(corr(conditioned_2d_corrblur(:,:),conditioned_2d_uncorr(:,:))));

    subplot(3,2,1);brain.drawPattern(brain.sampleNoisyVolume(ori));title('noisy pattern')
    subplot(3,2,3);brain.drawPattern(true_patterns(:,idx));title('true pattern')
    subplot(3,2,4);brain.drawPattern(clf2d.weights(1:400,idx));title('classifier weights')
    subplot(3,2,5);brain.drawPattern(mean(conditioned_2d_corrblur(:,1),2));title('correlated noise')
    subplot(3,2,6);brain.drawPattern(mean(conditioned_2d_uncorr(:,1),2));title('uncorrelated noise')
end

