---------------------------------------------------
-- various implementations for sampling position --
---------------------------------------------------
local M = {}

function M.uniform(sh, lh, sw, lw, obs, sprite)
  return torch.random(sh, lh), torch.random(sw, lw)
end

function M.split(sh, lh, sw, lw, obs, sprite)
  local function helper(small, large)
    local quarter = torch.ceil((large - small) / 8)
    local ranges = {
      {small, small + quarter},
      {small + quarter, large - quarter},
      {large - quarter, large}
    }
    local dice = torch.random(1,3)
   return torch.random(ranges[dice][1], ranges[dice][2])
  end
    
  return helper(sh, lh), helper(sw, lw)
end

function M.overlap_constraint(sh, lh, sw, lw, obs, sprite)
  local digit = sprite:ne(0)
  local digitArea = digit:sum()
    
  local function check(oh, ow, obs, sprite)
    local spriteH = sprite:size(2)
    local spriteW = sprite:size(3)
    local patch = obs[{{}, {oh, oh + spriteH - 1}, {ow, ow + spriteW - 1}}]:ne(0)       
    patch:cmul(digit)
    local overlapArea = patch:sum()
    if overlapArea / digitArea > 0.3 then
      return false
    else
      return true
    end
  end
  
  local oh, ow
  local count = 0
  repeat
    oh = torch.random(sh, lh)
    ow = torch.random(sw, lw)
    count = count + 1
    if count > 10 then return nil, nil end
  until check(oh, ow, obs, sprite)
  return oh, ow
end

function M.center(sh, lh, sw, lw, obs, sprite)
  local ch = math.ceil((obs:size(2) - sprite:size(2)) / 2)
  local cw = math.ceil((obs:size(3) - sprite:size(3)) / 2)

  local function inrange(x, sx, lx)
      return x >= sx and x <= lx
  end

  if inrange(ch, sh, lh) and inrange(cw, sw, lw) then
    return ch, cw
  else
    return nil, nil
  end
end

function M.sample(method, obs, sprite, border)
  assert(obs:dim() == 3, "expecting an image")
  assert(sprite:dim() == 3, "expecting a sprite")    
  local h = obs:size(2)
  local w = obs:size(3)
  local spriteH = sprite:size(2)
  local spriteW = sprite:size(3)
    
  local y, x = M[method](
    1 + border, h - spriteH + 1 - border,
    1 + border, w - spriteW + 1 - border,
    obs, sprite        
  )
  return y, x
end

return M
