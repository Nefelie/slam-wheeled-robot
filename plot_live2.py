
def update_mm_plot(ax, fig, N_mm_log, E_mm_log, N_gt_log, E_gt_log, x_coords, y_coords):
    # Update the plot with new data
    ax.clear()
    ax.plot(E_mm_log, N_mm_log, label="Motion Model", color='blue', linestyle="-")
    ax.plot(E_gt_log, N_gt_log, label="Ground Truth", color='black', linestyle="-")
    ax.scatter(x_coords, y_coords, color='green', marker='o')

    ax.legend()
    ax.set_xlabel('E')
    ax.set_ylabel('N')
    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-1.1,1.1)
    ax.set_aspect("equal")
    fig.canvas.draw()



def update_mm_plot2(ax, fig, x_coords, y_coords):
    ax.clear()
    ax.scatter(x_coords, y_coords, color='blue', marker='o')
    ax.set_xlabel('E')
    ax.set_ylabel('N')
    fig.canvas.draw()
