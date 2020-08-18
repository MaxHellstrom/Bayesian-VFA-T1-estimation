function plot_input_data(FA, Y, mask, ref, B1_corr)
% Function for plotting input data, prior to estimation. An example input
% dataset is available in the /data folder

figure('Name', 'Input data', 'Position', [0.5,0.5,1200,480])

for i=1:numel(FA)
    subplot(2,5,i)
    imagesc(Y(:,:,i),[0,5000]);
    if i==1
        ylabel('SPGR signal')
    end
    colorbar
    title(['degrees:', num2str(FA(i))])
    xticks([])
    yticks([])
    
    subplot(2,5,6)
    imagesc(mask)
    title('mask')
    colormap(gca, [0 0 0; 1 1 1]);
    colorbar('Ticks', [0, 1]);
    xticks([])
    yticks([])
    
    subplot(2,5,7)
    imagesc(B1_corr)
    colorbar
    title('B1 corr')
    xticks([])
    yticks([])
    
    subplot(2,5,9)
    imagesc(ref(:,:,1))
    colorbar
    title('PD ref')
    xticks([])
    yticks([])
    
    
    subplot(2,5,10)
    imagesc(ref(:,:,2))
    colorbar
    title('T1 ref')
    xticks([])
    yticks([])
end

end
