% %%%%%%%%%%%%%%
% the implement of KDLOR by alrash
% 
% paper
%   https://link.springer.com/article/10.1007/s11063-014-9340-2
%   sun at el. Constructing and Combining Orthogonal Projection Vectors for Ordinal Regression. NPL, 2014, vol 41, pp 139-155.
% %%%%%%%%%%%%%%
function [W] = kdlor_orth(data, label, C, K, p)
    d = size(data, 1); 
    center = zeros(d, K); m_distance_matrix = zeros(d, K - 1);
    Sw = zeros(d, d); W = zeros(d, p);

    for k = 1:K
        class_k = data(:, label == k); center(:, k) = mean(class_k, 2);

        if k > 1
            m_distance_matrix(:, k - 1) = center(:, k) - center(:, k - 1);
        end

        tmp = class_k - repmat(center(:, k), 1, size(class_k, 2));
        Sw = Sw + tmp * tmp';
    end
    
    Sw_inv = pinv(Sw);
    opts = optimset('Display','off');
    
    for i = 1:p
        H = [m_distance_matrix, W(:, 1:i - 1)]' * Sw_inv * [m_distance_matrix, W(:, 1:i - 1)];
        alpha = quadprog((H + H') / 2, [], [], [], [ones(1, K - 1), zeros(1, i - 1)], C, [zeros(K - 1, 1); -Inf * ones(i - 1, 1)], [], [], opts);
        W(:, i) = 1 / 2 * pinv(Sw) * [m_distance_matrix, W(:, 1:i - 1)] * alpha;
    end
end