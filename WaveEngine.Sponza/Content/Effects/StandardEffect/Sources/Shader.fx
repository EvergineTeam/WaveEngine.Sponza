[Begin-ResourceLayout]

[directives:IBL IBL_OFF IBL]
[directives:Emissive EMIS_OFF EMIS]
[directives:Specular SPEC_OFF SPEC]
[directives:Diffuse DIFF_OFF DIFF]
[directives:Normal NORMAL_OFF NORMAL]
[directives:AlphaTest ATEST_OFF ATEST]
[directives:Lighting LIT LIT_OFF]
[directives:DualTexture DUAL_OFF DUAL]
[directives:DualTextureType DUAL_LMAP DUAL_MUL DUAL_ADD DUAL_MSK]
[directives:Environment ENV_OFF ENV]
[directives:Fresnel FRES_OFF FRES]
[directives:VertexColor VCOLOR_OFF VCOLOR]

cbuffer PerDrawCall : register(b0)
{
	float4x4 	World			: packoffset(c0.x); [World]
	float4x4 	WorldInverseTranspose	: packoffset(c4.x); [WorldInverseTranspose]
};

cbuffer PerCamera : register(b1)
{
	float4x4	ViewProj[2]			: packoffset(c0.x); [StereoCameraViewProjection]
	float4		CameraPosition[2]		: packoffset(c8.x); [StereoCameraPosition]
	int		EyeCount			: packoffset(c10.x); [StereoEyeCount]
};

cbuffer Parameters : register(b2)
{
	float3	DiffuseColor		: packoffset(c0.x); [Default(1, 1, 1)]
	float	Alpha 				: packoffset(c0.w); [Default(1)]
	float3	AmbientColor		: packoffset(c1.x); [Default(0,0,0)]
	float	SpecularPower 		: packoffset(c1.w); [Default(64)]
	float3	EmissiveColor		: packoffset(c2.x); [Default(1,1,1)]
	float	FresnelFactor		: packoffset(c2.w); [Default(1)]
	float2	TextureOffset1		: packoffset(c3.x); 
	float2	TextureOffset2		: packoffset(c3.z);
	float	IBLFactor			: packoffset(c4.x);	[Default(1)]
	float	EnvironmentAmount 	: packoffset(c4.y); [Default(1)]
	float	ReferenceAlpha	 	: packoffset(c4.z); 
};

Texture2D DiffuseTexture	: register(t0);
SamplerState DiffuseSampler		: register(s0);

Texture2D DiffuseTexture2		: register(t1);
SamplerState DiffuseSampler2	: register(s1);

Texture2D NormalTexture			: register(t2);
SamplerState NormalSampler		: register(s2);

Texture2D EmissiveTexture		: register(t3);
SamplerState EmissiveSampler	: register(s3);

Texture2D SpecularTexture		: register(t4);
SamplerState SpecularSampler	: register(s4);

TextureCube IBLTexture			: register(t5);
SamplerState IBLSampler			: register(s5);

TextureCube EnvironmentTexture	: register(t6);
SamplerState EnvironmentSampler	: register(s6);

Texture2DArray LightingTexture : register(t7); [Lighting0]

[End-ResourceLayout]
[Begin-Pass:LPP_GBuffer]

	[profile 10_0]
	[requiredwith LIT]
	[entrypoints VS = VertexFunction PS = PixelFunction]

	struct VS_IN
	{
		float4 Position		: POSITION;
		float3 Normal		: NORMAL0;	
		uint   InstanceID	: SV_InstanceID;

	#if NORMAL
		float2 TexCoord		: TEXCOORD0;
		float3 Tangent		: TANGENT0;
	#endif
	};

	struct PS_IN
	{
		float4 Position 	: SV_POSITION;
		float4 PositionCS	: TEXCOORD0;
		float3 NormalWS		: TEXCOORD1;

	#if NORMAL
		float3 TangentWS	: TEXCOORD2;
		float3 BinormalWS	: TEXCOORD3;
		float2 TexCoord		: TEXCOORD4;
	#endif

		uint ArrayIndex : SV_RenderTargetArrayIndex;
	};

	struct PS_OUT
	{
		float4 Color 	: COLOR0;
		float4 Depth 	: COLOR1;
	};

	inline float3 EncodeNormalBase(float3 n)
	{
		return float3(n.xyz*0.5 + 0.5);
	}

	inline float4 EncodeNormalGlossiness(float3 normal, float glossiness)
	{
		float4 enc;
		enc.xyz = EncodeNormalBase(normal);
		enc.w = glossiness / 255;
		return enc;
	}

	PS_IN VertexFunction(VS_IN input)
	{
		PS_IN output = (PS_IN)0;

		int iid = input.InstanceID / EyeCount;
		int vid = input.InstanceID % EyeCount;

		float4x4 worldViewProj = mul(World, ViewProj[vid]);
		
		output.Position = mul(input.Position, worldViewProj);
		output.PositionCS = output.Position;
		output.NormalWS = mul(input.Normal, (float3x3)WorldInverseTranspose);

	#if NORMAL
		output.TangentWS = mul(input.Tangent, (float3x3)WorldInverseTranspose);
		output.BinormalWS = cross(output.TangentWS, output.NormalWS);
		output.TexCoord = input.TexCoord + TextureOffset1;
	#endif

		output.ArrayIndex = vid;

		return output;
	}

	PS_OUT PixelFunction(PS_IN input) : SV_Target
	{
		PS_OUT output = (PS_OUT)0;
		float3 normalWS;

	#if NORMAL
		// Normal Texture available
		// Normalize the tangent frame after interpolation
		float3x3 tangentFrameWS = float3x3(normalize(input.TangentWS),
		normalize(input.BinormalWS),
		normalize(input.NormalWS));

		// Sample the tangent-space normal map and decompress
		float3 normalTS = NormalTexture.Sample(NormalSampler, input.TexCoord).rgb;
		normalTS = normalize(normalTS * 2.0 - 1.0);

		// Convert to world space
		normalWS = mul(normalTS, tangentFrameWS);

	#else
		// No Normal texture defined
		normalWS = normalize(input.NormalWS);

	#endif 

		// Encode Normal and Specular power in a single RT (depth from Depth Buffer)
		// RT0: | N.x | N.y | N.z | Sp |
		output.Color = EncodeNormalGlossiness(normalWS, SpecularPower);
		output.Depth = input.PositionCS.z / input.PositionCS.w;
		
		return output;
	}
[End-Pass]

[Begin-Pass:Default]

[profile 10_0]
[entrypoints VS = VertexFunction PS = PixelFunction]

struct VS_IN
{
	float4 Position 	: POSITION;
	uint   InstanceID	: SV_InstanceID;		
#if DIFF || EMIS || SPEC || DUAL || ((ENV || IBL) && NORMAL)
	float2 TexCoord1 : TEXCOORD0;
#endif

#if DUAL
	float2 TexCoord2 : TEXCOORD1;
#endif

#if ENV || IBL
	float3 Normal	: NORMAL0;
	#if NORMAL
	float3 Tangent	: TANGENT0;
	#endif
#endif

#if VCOLOR
	float4 Color : COLOR0;
#endif
};

struct PS_IN
{
	float4 Position : SV_POSITION;

#if LIT
	float4 PositionCS : TEXCOORD0;
#endif

#if DIFF || EMIS || SPEC || NORMAL
	float2 TexCoord1 : TEXCOORD1;
#endif

#if DUAL
	float2 TexCoord2 : TEXCOORD2;
#endif

#if ENV
	float3 CameraVector : TEXCOORD3;
#endif

#if (ENV || IBL)
	float3 NormalWS	: TEXCOORD4;
	#if NORMAL
	float3 TangentWS	: TEXCOORD5;
	float3 BinormalWS	: TEXCOORD6;
	#endif
#endif

#if VCOLOR
	float4 Color : COLOR0;
#endif

	uint ArrayIndex : SV_RenderTargetArrayIndex;
};

inline float2 ComputeScreenPosition(float4 pos)
{
	float2 screenPos = pos.xy / pos.w;
	return (0.5f * (float2(screenPos.x, -screenPos.y) + 1));
}

inline void DecodeLightDiffuseSpecular(float4 enc, out float3 diffuse, out float specular)
{
	diffuse = enc.xyz;
	specular = enc.w;
	
	diffuse = diffuse * 2;	
}

inline float3 DecodeNormalBase(float3 enc)
{
	enc = enc*2-1;
	return normalize(enc);
}

inline void DecodeNormalGlossiness(float4 enc, out float3 normal, out float glossiness)
{
	normal = DecodeNormalBase(enc.xyz);
	glossiness = enc.w * 255;
}


PS_IN VertexFunction(VS_IN input)
{
	PS_IN output = (PS_IN)0;

	int iid = input.InstanceID / EyeCount;
	int vid = input.InstanceID % EyeCount;	

	float4x4 worldViewProj = mul(World, ViewProj[vid]);
	output.Position = mul(input.Position, worldViewProj);

#if LIT
	output.PositionCS = output.Position;
#endif

#if DIFF || EMIS || SPEC || NORMAL
	output.TexCoord1 = input.TexCoord1 + TextureOffset1;
#endif

#if DUAL
	output.TexCoord2 = input.TexCoord2 + TextureOffset2;
#endif

#if ENV || IBL
	output.NormalWS = mul(input.Normal, (float3x3)WorldInverseTranspose);

	#if ENV
	float3 positionWS = mul(input.Position, World).xyz;
	output.CameraVector = positionWS - CameraPosition[vid];
	#endif

	#if NORMAL
		output.TangentWS = mul(input.Tangent, (float3x3)WorldInverseTranspose);
		output.BinormalWS = cross(output.TangentWS, output.NormalWS);
	#endif
#endif

#if VCOLOR
	output.Color = input.Color;
#endif

	output.ArrayIndex = vid;

	return output;
}

float4 PixelFunction(PS_IN input) : SV_Target
{
	float3 diffuseIntensity = float3(1, 1, 1);
	float specularIntensity = 0;
	float3 basecolor = DiffuseColor;
	float3 intensity;
	float alphaMask = 1;

	#if VCOLOR
	basecolor *= input.Color.xyz;
	alphaMask *= input.Color.a;
#endif

#if LIT
	float2 screenPosition = ComputeScreenPosition(input.PositionCS);
	float4 lighting = LightingTexture.Sample(DiffuseSampler, float3(screenPosition, input.ArrayIndex));
	DecodeLightDiffuseSpecular(lighting, diffuseIntensity, specularIntensity);
#endif

#if DIFF
	float4 albedo = DiffuseTexture.Sample(DiffuseSampler, input.TexCoord1);
	alphaMask *= albedo.a;
	basecolor *= albedo.rgb;
#endif

#if DUAL
	float4 diffuse2 = DiffuseTexture2.Sample(DiffuseSampler2, input.TexCoord2);
	#if DUAL_LMAP
		basecolor *= 2 * diffuse2.rgb;
		alphaMask *= diffuse2.a;
	#endif

	#if DUAL_MUL
		basecolor *= diffuse2.rgb;
		alphaMask *= diffuse2.a;
	#endif

	#if DUAL_ADD
		float3 add = basecolor + diffuse2.rgb;
		basecolor = DiffuseColor * clamp(add, 0, 1);
		alphaMask += diffuse2.a;
	#endif

	#if DUAL_MSK
		basecolor = DiffuseColor * lerp(basecolor, diffuse2.rgb, diffuse2.a);
	#endif
#endif

#if ATEST
	if (alphaMask < ReferenceAlpha)
	{
		discard;
	}
#endif

#if SPEC
	float specular = SpecularTexture.Sample(SpecularSampler, input.TexCoord1).x;
	specularIntensity *= specular;
#endif

#if ENV || IBL

	float3 normalWS;

	#if NORMAL
		// Normal Texture available
		// Normalize the tangent frame after interpolation
		float3x3 tangentFrameWS = float3x3(
		normalize(input.TangentWS),
		normalize(input.BinormalWS),
		normalize(input.NormalWS));

		// Sample the tangent-space normal map and decompress
		float3 normalTS = NormalTexture.Sample(NormalSampler, input.TexCoord1).rgb;
		normalTS = normalize(normalTS * 2.0 - 1.0);

		// Convert to world space
		normalWS = mul(normalTS, tangentFrameWS);

	#else
		// No Normal texture defined
		normalWS = normalize(input.NormalWS);

	#endif 
#endif

#if IBL
	float3 ibl = IBLTexture.Sample(IBLSampler, normalWS).xyz;

	#if LIT
		diffuseIntensity += ibl * IBLFactor;
	#else
		diffuseIntensity *= ibl * IBLFactor;
	#endif
#endif

	float3 diffuseContribution = basecolor * diffuseIntensity;

#if ENV
	float3 cameraVector = normalize(input.CameraVector);
	float envAmount = EnvironmentAmount;

	#if FRES
		float fresnelTerm = abs(dot(cameraVector, normalWS));
		envAmount = pow(max(1 - fresnelTerm, 0), FresnelFactor) * EnvironmentAmount;
	#endif

	float3 envCoord = reflect(cameraVector, normalWS);
	float3 envColor = EnvironmentTexture.Sample(EnvironmentSampler, envCoord).rgb;

	#if SPEC
		envAmount *= specular;
	#endif

	diffuseContribution = lerp(diffuseContribution, envColor, envAmount);
#endif

	// Compute final color
	float3 color = diffuseContribution + specularIntensity + AmbientColor;

#if EMIS
	float3 emissive = EmissiveTexture.Sample(EmissiveSampler, input.TexCoord1).xyz;
	color += emissive * EmissiveColor;
#endif

	float4 finalColor = float4(color, alphaMask);
	finalColor *= Alpha;

	return finalColor;
}
[End-Pass]