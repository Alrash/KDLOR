% %%%%%%%%%%%%%%
% the implement of KDLOR by alrash
% 
% paper
%   https://ieeexplore.ieee.org/abstract/document/5184839
%   sun at el. Kernel Discriminant Learning for Ordinal Regression. IEEE TKDE, 2010, vol 22, pp 906-910.
% %%%%%%%%%%%%%%
function [w] = kdlor(data, label, C, K)
    d = size(data, 1); 
    center = zeros(d, K); m_distance_matrix = zeros(d, K - 1);
    Sw = zeros(d, d);

    for k = 1:K
        class_k = data(:, label == k); center(:, k) = mean(class_k, 2);

        if k > 1
            m_distance_matrix(:, k - 1) = center(:, k) - center(:, k - 1);
        end

        tmp = class_k - repmat(center(:, k), 1, size(class_k, 2));
        Sw = Sw + tmp * tmp';
    end
    
    Sw_inv = pinv(Sw);
    H = m_distance_matrix' * Sw_inv * m_distance_matrix;
    opts = optimset('Display','off');
    alpha = quadprog((H + H') / 2, [], [], [], ones(1, K - 1), C, zeros(K - 1, 1), [], [], opts);

    w = 1 / 2 * pinv(Sw) * m_distance_matrix * alpha;
end