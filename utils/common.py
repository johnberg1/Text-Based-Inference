from PIL import Image
import matplotlib.pyplot as plt


# Log images
def log_image(x, opts):
	return tensor2im(x)


def tensor2im(var):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def vis_faces(log_hooks, txt, mismatch_text):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(12, 4 * display_count))
	gs = fig.add_gridspec(display_count, 4)
	plt.gcf().text(0.02, 0.52, 'Original: {}'.format(txt[0]), fontsize=8)
	if mismatch_text:
		plt.gcf().text(0.02, 0.49, 'Mismatch: {}'.format(txt[-1]), fontsize=8)
		plt.gcf().text(0.02, 0.01, 'Mismatch: {}'.format(txt[0]), fontsize=8)
	else:
		plt.gcf().text(0.02, 0.49, 'Mismatch: {}'.format(txt[0]), fontsize=8)
		plt.gcf().text(0.02, 0.01, 'Mismatch: {}'.format(txt[1]), fontsize=8)
	plt.gcf().text(0.02, 0.04, 'Original: {}'.format(txt[1]), fontsize=8)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		vis_faces_with_age(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig
 
def vis_faces_two(log_hooks, txt, mismatch_text):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(12, 4 * display_count))
	gs = fig.add_gridspec(display_count, 4)
	#plt.gcf().text(0.02, 0.52, 'Original: {}'.format(txt[0]), fontsize=8)
	if mismatch_text:
		#plt.gcf().text(0.02, 0.49, 'Mismatch: {}'.format(txt[0]), fontsize=8)
		plt.gcf().text(0.02, 0.01, 'Mismatch: {}'.format(txt[0]), fontsize=8)
	else:
		#plt.gcf().text(0.02, 0.49, 'Mismatch: {}'.format(txt[0]), fontsize=8)
		plt.gcf().text(0.02, 0.01, 'Mismatch: {}'.format(txt[0]), fontsize=8)
	plt.gcf().text(0.02, 0.04, 'Original: {}'.format(txt[0]), fontsize=8)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		vis_faces_with_age(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig


def vis_faces_with_age(hooks_dict, fig, gs, i):
	fig.add_subplot(gs[i, 0])
	#print(hooks_dict.keys())
	plt.imshow(hooks_dict['input_face'])
	plt.title('Input\nOut Sim={:.2f}\n'.format(float(hooks_dict['diff_input'])))
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target\nIn={:.2f},Out={:.2f}\n'.format(float(hooks_dict['diff_views']),
	                                                                   float(hooks_dict['diff_target'])))
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output\nTarget Sim={:.2f}\n'.format(float(hooks_dict['diff_target'])))
	fig.add_subplot(gs[i, 3])
	plt.imshow(hooks_dict['recovered_face'])
	plt.title('Recovered\nTarget Sim={:.2f}\n'.format(float(hooks_dict['diff_target'])))
