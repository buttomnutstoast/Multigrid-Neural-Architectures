----------------------
-- Image Processing --
----------------------
require 'image'

local M = {}

function M.rescale(sprite, scale)
  -- rescale
  local large, small
  if scale[1] >= scale[2] then
    large, small = unpack(scale)
  else
    small, large = unpack(scale)
  end
  local s = torch.ceil(torch.uniform(small, large)*sprite:size(2))
  sprite = image.scale(sprite, s, s)
  return sprite
end

function M.rotate(sprite, angle)
  local theta = angle / 180 * math.pi
  local rot_theta = torch.uniform(-theta, theta)
  sprite = image.rotate(sprite, rot_theta)
  return sprite
end

function M.affine(sprite, deg)
  local theta = deg / 180 * math.pi
  local aff_theta = torch.uniform(-theta, theta)
  local tan = torch.tan(aff_theta)
  local diag = {tan, 0}
  local order = torch.randperm(2)
  local affineMat = torch.FloatTensor(
    {{1, diag[order[1]]}, {diag[order[2]], 1}}
  )
  sprite = image.affinetransform(sprite:double(), affineMat)
  return sprite
end


function M.normalize(sprite)
  local max = sprite:max()
  local ratio = 1 / max
  sprite:mul(ratio)
  return sprite
end

return M
