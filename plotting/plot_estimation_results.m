function plot_estimation_results(ML, B_unif, B_TV, ref, lim_T1, lim_error_T1, rotation_angle)
% Function for plotting the estimation results when using synthetic data with known
% reference value

rows=2; cols=3;
data = cell(rows, cols);

labelsize = 16;
mask = ML.model_params.mask;

data{1,1} = imrotate(ML.results.T1, rotation_angle);
data{1,2} = imrotate(mean(squeeze(B_unif.results.T1), 3), rotation_angle);
data{1,3} = imrotate(mean(squeeze(B_TV.results.T1), 3), rotation_angle);

data{2,1} = imrotate(100*mask.*(ML.results.T1-ref(:,:,2))./(ref(:,:,2)), rotation_angle);
data{2,2} = imrotate(100*mask.*(mean(squeeze(B_unif.results.T1),3)-ref(:,:,2))./(ref(:,:,2)), rotation_angle);
data{2,3} = imrotate(100*mask.*(mean(squeeze(B_TV.results.T1),3)-ref(:,:,2))./(ref(:,:,2)), rotation_angle);

data{2,1}(isnan(data{2,1}))=0;
data{2,2}(isnan(data{2,2}))=0;
data{2,3}(isnan(data{2,3}))=0;


r_i=28;

plot_width = 20;
plot_height = 18;

text_x_pos = 0.02;
text_y_pos = 0.9;



scale = (plot_width/plot_height);

indent_x = 0.04;
indent_y = 0.1;
indent_y = scale * indent_y;
side_x = 1/3 - indent_x/3;
side_y = scale * side_x;

plots = cell(2,3);
figure('Units', 'centimeters', 'Position', [0 0 plot_width plot_height])

height_ind = 10;

plots{1} = subplot(2,3,1);
imagesc(data{1,1}(r_i:end-r_i,r_i:end-r_i), lim_T1);
xticks([])
yticks([])
colormap(gray(256))
text(text_x_pos,text_y_pos,['(a)'],'FontSize',labelsize,'FontWeight','Bold','Color',[1, 1, 1], 'Units','normalized')
title('ML')

cb_T1 = colorbar('southoutside');
cb_T1.TickLabelInterpreter = 'Latex';
cb_T1.Visible='on';
colormap(gray(256))
xticks([])
yticks([])
cb_T1.Label.Interpreter = 'latex';
cb_T1.Label.String = '$\widehat{T}_{1}$ [s]';

plots{2} = subplot(2,3,2);
imagesc(data{1,2}(r_i:end-r_i,r_i:end-r_i), lim_T1);
xticks([])
yticks([])
colormap(gray(256))
text(text_x_pos,text_y_pos,['(b)'],'FontSize',labelsize,'FontWeight','Bold','Color',[1, 1, 1], 'Units','normalized')
title('$\mathrm{B}_{\mathrm{unif}}$')

plots{3} = subplot(2,3,3);
imagesc(data{1,3}(r_i:end-r_i,r_i:end-r_i), lim_T1);
xticks([])
yticks([])
colormap(gray(256))
text(text_x_pos,text_y_pos,['(c)'],'FontSize',labelsize,'FontWeight','Bold','Color',[1, 1, 1], 'Units','normalized')
title('$\mathrm{B}_{\mathrm{TV}}$')

plots{4} = subplot(2,3,4);
img4 = data{2,1};
imagesc(img4(r_i:end-r_i,r_i:end-r_i), lim_error_T1);
xticks([])
yticks([])
colormap(gca, redblue(256))
text(text_x_pos,text_y_pos,['(d)'],'FontSize',labelsize,'FontWeight','Bold','Color',[0, 0, 0], 'Units','normalized')

cb_T1_error = colorbar('southoutside');
cb_T1_error.TickLabelInterpreter = 'Latex';
cb_T1_error.Visible='on';
xticks([])
yticks([])
cb_T1_error.Label.Interpreter = 'latex';
cb_T1_error.Label.String = '$T_{1}$ error [\%]';


plots{5} = subplot(2,3,5);
img5 = data{2,2};
imagesc(img5(r_i:end-r_i,r_i:end-r_i), lim_error_T1);
xticks([])
yticks([])
colormap(gca, redblue(256))
text(text_x_pos,text_y_pos,['(e)'],'FontSize',labelsize,'FontWeight','Bold','Color',[0, 0, 0], 'Units','normalized')


plots{6} = subplot(2,3,6);
img6 = data{2,3};
imagesc(img6(r_i:end-r_i,r_i:end-r_i), lim_error_T1);
xticks([])
yticks([])
colormap(gca, redblue(256))
text(text_x_pos,text_y_pos, ['(f)'],'FontSize',labelsize,'FontWeight','Bold','Color',[0, 0, 0], 'Units','normalized')


h_ind = 0.04;
plots{1}.Position = [indent_x+0*side_x, 1.4*side_y + indent_y, side_x, side_y];
plots{2}.Position = [indent_x+1*side_x, 1.4*side_y + indent_y, side_x, side_y];
plots{3}.Position = [indent_x+2*side_x, 1.4*side_y + indent_y, side_x, side_y];
plots{4}.Position = [indent_x+0*side_x+0*h_ind, 0*side_y + indent_y, side_x, side_y];
plots{5}.Position = [indent_x+1*side_x+0*h_ind, 0*side_y + indent_y, side_x, side_y];
plots{6}.Position = [indent_x+2*side_x+0*h_ind, 0*side_y + indent_y, side_x, side_y];

cb_T1.Position(3)=3*side_x;
cb_T1_error.Position(3)=3*side_x;

sbar = 0.035;

cb_T1.Position(3) = cb_T1.Position(3) -sbar;
cb_T1.Position(1) = cb_T1.Position(1) + 0.5* sbar;
cb_T1_error.Position(3) = cb_T1_error.Position(3) - sbar;
cb_T1_error.Position(1) = cb_T1_error.Position(1) + 0.5* sbar;

end
