classdef Simfeed2dBrain < handle
    properties
    VOXEL_SIZE = 3 % in mm
    VOXEL_DIM = 20 % number of voxels, NxN
    NOISE_FWHM = 3 %5 % in mm
    neural_voxels
    noise_std = 0.5
    CORR_RATIO = 0.5 % noise correlation ratio
    noise_gauss_kernel = 5 % in mm
    TUNED_VOXELS = 0.5 % percent
    orientation_tunings
    orientation_magnitudes
    voxel_mapping
    conditioned_activity
    learning_rate = 0.1
    end

    methods
        function self = Simfeed2dBrain()
            self.generateVoxelMappings;
            self.resetToBaseline;
        end

        function resetToBaseline(self)
            tuned_voxel_count = self.TUNED_VOXELS*self.VOXEL_DIM^2;
            untuned_voxel_count = (1-self.TUNED_VOXELS)*self.VOXEL_DIM^2;
            self.conditioned_activity = zeros(self.VOXEL_DIM^2,1);
            self.orientation_tunings = repmat([0:22.5:157.5]',tuned_voxel_count/8,1);
            self.orientation_magnitudes = ones(tuned_voxel_count,1);
            self.conditioned_activity = zeros(self.VOXEL_DIM^2,1);
        end

        function setNoise(self, level)
            self.noise_std = level;
        end

        function noise = generateNewNoise(self)
            uncorr_noise = randn(self.VOXEL_DIM);
            sigma_unscaled = self.NOISE_FWHM/2.355;
            sigma_scaled = sigma_unscaled/self.VOXEL_SIZE;
            noise = imgaussfilt(uncorr_noise,sigma_scaled);
            noise = self.noise_std*noise(:);
        end

        function noise = generateNewNoiseCorr(self)
            noise = self.CORR_RATIO*self.sampleVolume(180*rand) + (1-self.CORR_RATIO)*self.generateNewNoise();
        end

        function noise = generateNewNoiseCorrBlur(self)
            uncorr_noise = randn(self.VOXEL_DIM);
            sigma_unscaled = self.NOISE_FWHM/2.355;
            sigma_scaled = sigma_unscaled/self.VOXEL_SIZE;
            raw_noise = self.noise_std*uncorr_noise;
            total_noise = self.CORR_RATIO*reshape(self.sampleVolume(180*rand),self.VOXEL_DIM,self.VOXEL_DIM)+(1-self.CORR_RATIO)*raw_noise;
            noise = imgaussfilt(total_noise,sigma_scaled);
            noise = noise(:);
        end

        function reinforcementLearn(self, conditioned_pattern, score_change)
            self.conditioned_activity = self.conditioned_activity + score_change*self.learning_rate*conditioned_pattern;
        end

        function generateVoxelMappings(self)
            tuned_voxel_count = self.TUNED_VOXELS*self.VOXEL_DIM^2;
            untuned_voxel_count = (1-self.TUNED_VOXELS)*self.VOXEL_DIM^2;
            voxel_labels = [(1:tuned_voxel_count)'; zeros(untuned_voxel_count,1)]; 
            rng(2);
            voxel_labels = Shuffle(voxel_labels);
            rng('shuffle');
            self.voxel_mapping = zeros(tuned_voxel_count,self.VOXEL_DIM^2);
            for tuning = 1:tuned_voxel_count
                for voxel = 1:self.VOXEL_DIM^2
                    if voxel_labels(voxel) == tuning
                        self.voxel_mapping(tuning,voxel) = 1;
                    end
                end
            end
        end

        function receptor_tunings = viewOrientation(self, orientation)
            shifted_orientations = self.wrapAngle(self.orientation_tunings-self.wrapAngle(orientation));
            receptor_tunings = self.orientation_magnitudes.*self.calcTuning(shifted_orientations);
        end

        function induced_activity = sampleVolume(self, viewed_orientation)
            induced_activity = self.voxel_mapping'*self.viewOrientation(viewed_orientation);
        end

        function noisy_induced_activity = sampleNoisyVolume(self, viewed_orientation)
            noisy_induced_activity = (self.voxel_mapping'*self.viewOrientation(viewed_orientation) + self.generateNewNoise);
        end

        function noisy_induced_activity = sampleNoisyCorrVolume(self, viewed_orientation)
            noisy_induced_activity = (self.voxel_mapping'*self.viewOrientation(viewed_orientation) + self.generateNewNoiseCorr);
        end

        function noisy_induced_activity = sampleNoisyCorrBlurVolume(self, viewed_orientation)
            noisy_induced_activity = (self.voxel_mapping'*self.viewOrientation(viewed_orientation) + self.generateNewNoiseCorrBlur);
        end

        function conditioned_induced_activity = sampleNoisyConditionedVolume(self)
            conditioned_induced_activity = self.conditioned_activity + self.generateNewNoise;
        end

        function conditioned_induced_activity = sampleNoisyCorrConditionedVolume(self)
            conditioned_induced_activity = self.conditioned_activity + self.generateNewNoiseCorr;
        end

        function conditioned_induced_activity = sampleNoisyCorrBlurConditionedVolume(self)
            conditioned_induced_activity = self.conditioned_activity + self.generateNewNoiseCorrBlur;
        end

        function drawPattern(self,pattern)
            crange = 1.5;
            oversamplerate = 5;
            imagesc(reshape(pattern,self.VOXEL_DIM,self.VOXEL_DIM));
            lowrescmap = redbluecmap;
            hirescmap = resample(lowrescmap,oversamplerate,1);
            hirescmap = max(hirescmap(1:(size(hirescmap,1)-oversamplerate),:),0);
            colormap(hirescmap);
            caxis([-crange, crange]);
            % colorbar;
        end
    end

    methods(Static)
        function tuning = calcTuning(orientation)
            tuning = exp(log(2)/22.5*(abs((orientation)-90)-90));
        end

        function wrapped_orientation = wrapAngle(orientation)
            wrapped_orientation = orientation - 180*floor(orientation/180);
        end
    end
end
