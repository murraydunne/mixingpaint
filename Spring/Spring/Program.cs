using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Drawing.Imaging;
using Cudafy;
using System.IO;

namespace Spring
{
    class Program
    {
        private const int width = 512;
        private const int height = 512;
        private const int numParticles = 10 * 10 * 10 * 10;
        private const int numParticleCyclesPerFrame = 10;
        private const int sortBufferLength = 100;

        private const int numberOfSeconds = 30;
        private const int framesPerSecond = 30;

        private const float deltaTimePerParticleCycle = (1.0f / (float)framesPerSecond) / (float)numParticleCyclesPerFrame;

        private const float particleMass = 0.01f;
        private const float accelerationDueToGravity = 0.5f;
        
        private const float knifeRenderWidthSquared = 21.16f; // was 19.36f;
        private const float knifeRenderVerticalOffsetFromTime = 0.0f; // was 1.8f;

        private const float color1R = 0.78f;
        private const float color1G = 0.0f;
        private const float color1B = 1.0f;

        private const float color2R = 0.51f;
        private const float color2G = 0.86f;
        private const float color2B = 0.95f;

        private static bool hasRestDensityBeenComputed = false;
        private static float restDensity = 0.0f;

        private static float[] pX1;
        private static float[] pX2;
        private static float[] pY1;
        private static float[] pY2;
        private static float[] pZ1;
        private static float[] pZ2;

        private static float[] velocityX;
        private static float[] velocityY;
        private static float[] velocityZ;

        private static float[] colorPosition;
        private static float[] colorVelocity;

        private static float[] currentTime;

        static void Main(string[] args)
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            gpu.LoadModule(km);

            int numFrames = numberOfSeconds * framesPerSecond;
            InitializeParticles();

            File.WriteAllText("length.txt", numFrames.ToString());

            for(int i = 0; i < numFrames; i++)
            {
                DateTime frameStart = DateTime.Now;

                Simulate(gpu);
                Bitmap frame = Render(gpu, i);

                TimeSpan frameTime = DateTime.Now - frameStart;
                Console.WriteLine("Frame " + i + " complete. Time: " + frameTime.TotalMilliseconds + "ms");
            }
        }

        static Bitmap Render(GPGPU gpu, int frameNum)
        {
            uint[,] deviceImage = gpu.Allocate<uint>(width, height);

            float[] pX1_gpu = gpu.CopyToDevice<float>(pX1);
            float[] pY1_gpu = gpu.CopyToDevice<float>(pY1);
            float[] pZ1_gpu = gpu.CopyToDevice<float>(pZ1);
            
            float[] colorPosition_gpu = gpu.CopyToDevice<float>(colorPosition);
            float[] currentTime_gpu = gpu.CopyToDevice<float>(currentTime);

            dim3 threadsPerBlock = new dim3(8, 8);
            dim3 numBlocks = new dim3(width / threadsPerBlock.x, height / threadsPerBlock.y);

            gpu.Launch(numBlocks, threadsPerBlock).renderKernel(deviceImage, pX1_gpu, pY1_gpu, pZ1_gpu, colorPosition_gpu, currentTime_gpu);

            uint[,] finalImage = new uint[width, height];
            gpu.CopyFromDevice<uint>(deviceImage, finalImage);

            gpu.Free(deviceImage);
            gpu.Free(pX1_gpu);
            gpu.Free(pY1_gpu);
            gpu.Free(pZ1_gpu);

            gpu.Free(colorPosition_gpu);
            gpu.Free(currentTime_gpu);

            GCHandle pixels = GCHandle.Alloc(finalImage, GCHandleType.Pinned);
            Bitmap bmp = new Bitmap(width, height, width * sizeof(int), PixelFormat.Format32bppRgb, pixels.AddrOfPinnedObject());
            bmp.Save("spring" + frameNum + ".png");
            pixels.Free();

            return bmp;
        }

        static void InitializeParticles()
        {
            pX1 = new float[numParticles];
            pX2 = new float[numParticles];
            pY1 = new float[numParticles];
            pY2 = new float[numParticles];
            pZ1 = new float[numParticles];
            pZ2 = new float[numParticles];

            velocityX = new float[numParticles];
            velocityY = new float[numParticles];
            velocityZ = new float[numParticles];

            colorPosition = new float[numParticles];
            colorVelocity = new float[numParticles];

            currentTime = new float[1];
            currentTime[0] = 0.0f;

            Random rand = new Random();
            for (int i = 0; i < numParticles; i++)
            {
                pX1[i] = (float)rand.NextDouble() * 3.0f - 1.5f;
                pY1[i] = (float)rand.NextDouble() * 3.0f - 1.5f;
                pZ1[i] = (float)rand.NextDouble() * 3.0f - 1.5f;

                colorVelocity[i] = 0.0f;

                if (pY1[i] < 0.0f)
                {
                    pY1[i] -= 0.5f;
                    colorPosition[i] = 0.0f;
                }
                else
                {
                    pY1[i] += 0.5f;
                    colorPosition[i] = 1.0f;
                }
            }
        }

        static void Simulate(GPGPU gpu)
        {
            float[] pX1_gpu = gpu.CopyToDevice<float>(pX1);
            float[] pX2_gpu = gpu.CopyToDevice<float>(pX2);
            float[] pY1_gpu = gpu.CopyToDevice<float>(pY1);
            float[] pY2_gpu = gpu.CopyToDevice<float>(pY2);
            float[] pZ1_gpu = gpu.CopyToDevice<float>(pZ1);
            float[] pZ2_gpu = gpu.CopyToDevice<float>(pZ2);

            float[] density_gpu = gpu.Allocate<float>(numParticles);

            float[] velocityX_gpu = gpu.CopyToDevice<float>(velocityX);
            float[] velocityY_gpu = gpu.CopyToDevice<float>(velocityY);
            float[] velocityZ_gpu = gpu.CopyToDevice<float>(velocityZ);

            float[] colorPosition1_gpu = gpu.CopyToDevice<float>(colorPosition);
            float[] colorVelocity1_gpu = gpu.CopyToDevice<float>(colorVelocity);
            float[] colorPosition2_gpu = gpu.Allocate<float>(numParticles);
            float[] colorVelocity2_gpu = gpu.Allocate<float>(numParticles);

            dim3 threadsPerBlock = new dim3(10, 10);
            dim3 numBlocks = new dim3(10, 10);
            
            if (!hasRestDensityBeenComputed)
            {
                gpu.Launch(numBlocks, threadsPerBlock).densityKernel(pX1_gpu, pY1_gpu, pZ1_gpu, density_gpu);

                float[] densityCpu = new float[numParticles];
                gpu.CopyFromDevice<float>(density_gpu, densityCpu);
                restDensity = densityCpu.Average();
                hasRestDensityBeenComputed = true;
            }

            float[] restDensityCpuArray = new float[] { restDensity };
            float[] restDensity_gpu = gpu.CopyToDevice<float>(restDensityCpuArray);

            float[] currentTime_gpu = null;

            for (int i = 0; i < numParticleCyclesPerFrame / 2; i++)
            {
                currentTime_gpu = gpu.CopyToDevice<float>(currentTime);

                gpu.Launch(numBlocks, threadsPerBlock).densityKernel(pX1_gpu, pY1_gpu, pZ1_gpu, density_gpu);
                gpu.Launch(numBlocks, threadsPerBlock).particleKernel(pX1_gpu, pY1_gpu, pZ1_gpu, pX2_gpu, pY2_gpu, pZ2_gpu, 
                    velocityX_gpu, velocityY_gpu, velocityZ_gpu, density_gpu, restDensity_gpu, 
                    colorPosition1_gpu, colorVelocity1_gpu, colorPosition2_gpu, colorVelocity2_gpu, currentTime_gpu);

                gpu.CopyFromDevice<float>(currentTime_gpu, currentTime);
                gpu.Free(currentTime_gpu);
                currentTime[0] += deltaTimePerParticleCycle;
                currentTime_gpu = gpu.CopyToDevice<float>(currentTime);

                gpu.Launch(numBlocks, threadsPerBlock).densityKernel(pX2_gpu, pY2_gpu, pZ2_gpu, density_gpu);
                gpu.Launch(numBlocks, threadsPerBlock).particleKernel(pX2_gpu, pY2_gpu, pZ2_gpu, pX1_gpu, pY1_gpu, pZ1_gpu, 
                    velocityX_gpu, velocityY_gpu, velocityZ_gpu, density_gpu, restDensity_gpu,
                    colorPosition2_gpu, colorVelocity2_gpu, colorPosition1_gpu, colorVelocity1_gpu, currentTime_gpu);
                
                gpu.CopyFromDevice<float>(currentTime_gpu, currentTime);
                gpu.Free(currentTime_gpu);
                currentTime[0] += deltaTimePerParticleCycle;
            }

            gpu.CopyFromDevice<float>(pX1_gpu, pX1);
            gpu.CopyFromDevice<float>(pY1_gpu, pY1);
            gpu.CopyFromDevice<float>(pZ1_gpu, pZ1);
            
            gpu.CopyFromDevice<float>(velocityX_gpu, velocityX);
            gpu.CopyFromDevice<float>(velocityY_gpu, velocityY);
            gpu.CopyFromDevice<float>(velocityY_gpu, velocityY);
            
            gpu.CopyFromDevice<float>(colorPosition1_gpu, colorPosition);
            gpu.CopyFromDevice<float>(colorVelocity1_gpu, colorVelocity);

            gpu.Free(restDensity_gpu);
            gpu.Free(pX1_gpu);
            gpu.Free(pY1_gpu);
            gpu.Free(pZ1_gpu);
            gpu.Free(pX2_gpu);
            gpu.Free(pY2_gpu);
            gpu.Free(pZ2_gpu);

            gpu.Free(density_gpu);

            gpu.Free(velocityX_gpu);
            gpu.Free(velocityY_gpu);
            gpu.Free(velocityZ_gpu);

            gpu.Free(colorPosition1_gpu);
            gpu.Free(colorVelocity1_gpu);
            gpu.Free(colorPosition2_gpu);
            gpu.Free(colorVelocity2_gpu);
        }

        [Cudafy]
        private static float KernelPolynomial(float deltaR, float h)
        {
            if (deltaR <= 0.0f || deltaR > h)
            {
                return 0.0f;
            }
            float squaredTerm = ((h * h) - (deltaR * deltaR));
            return (315.0f * squaredTerm * squaredTerm * squaredTerm) / (64.0f * GMath.PI * GMath.Pow(h, 9.0f));
        }

        [Cudafy]
        private static float KernelGradientPolynomial(float deltaR, float h)
        {
            if (deltaR <= 0.0f || deltaR > h)
            {
                return 0.0f;
            }

            float squaredTerm = ((h * h) - (deltaR * deltaR));
            return -(945.0f * deltaR * squaredTerm * squaredTerm) / (32.0f * GMath.PI * GMath.Pow(h, 9.0f));
        }

        [Cudafy]
        private static float KernelDoubleGradientPolynomial(float deltaR, float h)
        {
            if (deltaR <= 0.0f || deltaR > h)
            {
                return 0.0f;
            }

            //float squaredTerm = ((h * h) - (deltaR * deltaR));
            //float squaredTermWithFive = ((h * h) - (5.0f * deltaR * deltaR));

            //return -(945.0f * squaredTerm * squaredTermWithFive) / (32.0f * GMath.PI * GMath.Pow(h, 9.0f));

            float squaredTerm = ((h * h) - (deltaR * deltaR));
            float squaredTermWithCoeffs = ((3.0f * h * h) - (7.0f * deltaR * deltaR));

            return -(945.0f * squaredTerm * squaredTermWithCoeffs) / (32.0f * GMath.PI * GMath.Pow(h, 9.0f));
        }

        [Cudafy]
        private static float KernelGradientSpiky(float deltaR, float h)
        {
            if (deltaR <= 0.0f || deltaR > h)
            {
                return 0.0f;
            }

            float squaredTerm = (h - deltaR);
            return -(45.0f * squaredTerm * squaredTerm) / (GMath.PI * GMath.Pow(h, 6.0f));
        }

        [Cudafy]
        private static float KernelDoubleGradientViscosity(float deltaR, float h)
        {
            if (deltaR <= 0.0f || deltaR > h)
            {
                return 0.0f;
            }

            //float term1 = h * h * h * h;
            //float term2 = 2.0f * h * deltaR * deltaR * deltaR;
            //float term3 = -3.0f * deltaR * deltaR * deltaR * deltaR;
            //float numerator = 15.0f * (term1 + term2 + term3);
            //float denominator = 2.0f * GMath.PI * GMath.Pow(h, 6.0f) * deltaR * deltaR * deltaR;

            //return numerator / denominator;
            
            return (45.0f * (h - deltaR)) / (GMath.PI * GMath.Pow(h, 6.0f));
        }

        [Cudafy]
        public static void densityKernel(GThread thread, float[] pX, float[] pY, float[] pZ, float[] density)
        {
            int particleIndex = thread.blockIdx.x + thread.blockIdx.y * 10 + thread.threadIdx.x * 10 * 10 + thread.threadIdx.y * 10 * 10 * 10;
            Vector3 currentParticle = new Vector3(pX[particleIndex], pY[particleIndex], pZ[particleIndex]);

            float particleDensity = 0.0f;
            float densityKernelH = 0.5f;

            for (int i = 0; i < pX.Length; i++)
            {
                Vector3 otherParticle = new Vector3(pX[i], pY[i], pZ[i]);
                Vector3 deltaR = currentParticle.Minus(otherParticle);
                
                particleDensity += particleMass * KernelPolynomial(deltaR.Magnitude(), densityKernelH);
            }

            density[particleIndex] = particleDensity;
        }

        [Cudafy]
        public static float KnifeXOffsetAtTime(float time)
        {
            if(time < 2.0f)
            {
                return -1.5f + (time * 1.65f);
            }

            return 1.8f;
        }

        [Cudafy]
        public static void particleKernel(GThread thread, float[] pX, float[] pY, float[] pZ, float[] pXnew, float[] pYnew, float[] pZnew, 
            float[] vX, float[] vY, float[] vZ, float[] density, float[] restDensityArray, 
            float[] colorPosition, float[] colorVelocity, float[] newColorPosition, float[] newColorVelocity, float[] currentTimeArray)
        {
            int particleIndex = thread.blockIdx.x + thread.blockIdx.y * 10 + thread.threadIdx.x * 10 * 10 + thread.threadIdx.y * 10 * 10 * 10;
            Vector3 currentParticlePosition = new Vector3(pX[particleIndex], pY[particleIndex], pZ[particleIndex]);
            Vector3 currentParticleVelocity = new Vector3(vX[particleIndex], vY[particleIndex], vZ[particleIndex]);

            // the current time
            float currentTime = currentTimeArray[0];

            // the rest density
            float restDensity = restDensityArray[0];

            // force of pressure setup
            float idealGasConstantForThisFluid = 1.8615f; //0.4615f; // <- the actual value of water
            float pressureKernelH = 0.5f;
            Vector3 particleForceOfPressure = new Vector3(0.0f, 0.0f, 0.0f);
            float particlePressure = idealGasConstantForThisFluid * (density[particleIndex] - restDensity);

            // force of viscosity setup
            float fluidViscosity = 0.4f;
            float viscosityKernelH = 0.5f;
            Vector3 particleForceOfViscosity = new Vector3(0.0f, 0.0f, 0.0f);

            // force of surface tension setup
            float fluidSurfaceTension = 0.1f; // 0.4f;
            Vector3 smoothedColorField = new Vector3(0.0f, 0.0f, 0.0f);
            float divergenceOfSmoothedColorField = 0.0f;
            float surfaceTensionKernelH = 0.5f;
            Vector3 particleForceOfSurfaceTension = new Vector3(0.0f, 0.0f, 0.0f);

            // color setup
            float colorAccelerationCoefficient = 0.1f;
            float colorSumReductionCoefficient = 0.001f;
            float forceOfColor = 0.0f;
            float currentParticleColor = colorPosition[particleIndex];
            float colorKernelH = 0.1f;

            for (int i = 0; i < pX.Length; i++)
            {
                if(i != particleIndex) // don't compare to yourself
                {
                    Vector3 otherParticle = new Vector3(pX[i], pY[i], pZ[i]);
                    Vector3 otherParticleVelocity = new Vector3(vX[i], vY[i], vZ[i]);
                    Vector3 deltaR = currentParticlePosition.Minus(otherParticle);
                    float deltaRMagnitude = deltaR.Magnitude();
                    Vector3 deltaRNormalized = deltaR.Normalized();

                    if(deltaR.Magnitude() > 0.00000000000001f && density[i] > 0.00000000000001f)
                    {
                        // do pressure
                        float otherParticlePressure = idealGasConstantForThisFluid * (density[i] - restDensity);
                        float massPressureTerm = particleMass * ((otherParticlePressure + particlePressure) / (density[i] * 2.0f));
                        float magnitudeOfForceOfPressure = massPressureTerm * KernelGradientSpiky(deltaRMagnitude, pressureKernelH);

                        particleForceOfPressure = particleForceOfPressure.Minus(deltaRNormalized.Multiply(magnitudeOfForceOfPressure));

                        // do viscosity
                        Vector3 massVelocityTerm = currentParticleVelocity.Minus(otherParticleVelocity).Divide(density[i]).Multiply(particleMass);
                        float kernelTerm = KernelDoubleGradientViscosity(deltaRMagnitude, viscosityKernelH);

                        particleForceOfViscosity = particleForceOfViscosity.Minus(massVelocityTerm.Multiply(kernelTerm));

                        // do surface tension
                        float smoothedColorScalar = particleMass * (1.0f / density[i]) * KernelGradientPolynomial(deltaRMagnitude, surfaceTensionKernelH);
                        smoothedColorField = smoothedColorField.Minus(deltaRNormalized.Multiply(smoothedColorScalar));

                        float divergenceOfSmoothedColorScalar = particleMass * (1.0f / density[i]) * 
                            KernelDoubleGradientPolynomial(deltaRMagnitude, surfaceTensionKernelH);
                        divergenceOfSmoothedColorField -= divergenceOfSmoothedColorField;

                        // do color
                        float otherColor = colorPosition[i];
                        float rangedOurColor = currentParticleColor - 0.5f;
                        float rangedOtherColor = otherColor - 0.5f;

                        float colorTerm = (rangedOtherColor * rangedOtherColor * rangedOtherColor - rangedOurColor * rangedOurColor * rangedOurColor);
                        forceOfColor += colorTerm * KernelPolynomial(deltaRMagnitude, colorKernelH);
                    }
                }
            }

            // final viscosity math
            particleForceOfViscosity = particleForceOfViscosity.Multiply(fluidViscosity);

            // final surface tension math
            float smoothedColorFieldMagnitude = smoothedColorField.Magnitude();
            if (smoothedColorFieldMagnitude > 0.00000000000001f)
            {
                particleForceOfSurfaceTension = smoothedColorField.Multiply(divergenceOfSmoothedColorField)
                    .Divide(smoothedColorFieldMagnitude).Multiply(-fluidSurfaceTension);
            }

            // final gravity math
            Vector3 particleForceOfGravity = new Vector3(1.0f, 0.0f, 0.0f).Multiply(density[particleIndex] * accelerationDueToGravity);

            // final color math
            float colorAcceleration = (forceOfColor / density[particleIndex]) * colorAccelerationCoefficient;
            newColorPosition[particleIndex] = currentParticleColor + colorVelocity[particleIndex] * deltaTimePerParticleCycle +
                0.5f * colorAcceleration * deltaTimePerParticleCycle * deltaTimePerParticleCycle;
            newColorVelocity[particleIndex] = colorVelocity[particleIndex] + colorAcceleration * deltaTimePerParticleCycle;

            if (newColorPosition[particleIndex] < 0.0f)
            {
                newColorPosition[particleIndex] = 0.0f;
            }
            else if (newColorPosition[particleIndex] > 1.0f)
            {
                newColorPosition[particleIndex] = 1.0f;
            }

            if (newColorVelocity[particleIndex] < 0.0f)
            {
                newColorVelocity[particleIndex] = 0.0f;
            }
            else if (newColorVelocity[particleIndex] > 1.0f)
            {
                newColorVelocity[particleIndex] = 1.0f;
            }

            // sum all forces on particle
            Vector3 sumOfForces = particleForceOfPressure.Plus(particleForceOfGravity).Plus(particleForceOfViscosity)
                .Plus(particleForceOfSurfaceTension);
            Vector3 sumOfAcceleration = new Vector3(0, 0, 0);

            if (density[particleIndex] > 0.00000000000001f)
            {
                sumOfAcceleration = sumOfForces.Divide(density[particleIndex]);
            }

            Vector3 positionDelta = currentParticleVelocity.Multiply(deltaTimePerParticleCycle).Plus(
                sumOfAcceleration.Multiply(deltaTimePerParticleCycle * deltaTimePerParticleCycle * 0.5f));
            Vector3 velocityDelta = sumOfAcceleration.Multiply(deltaTimePerParticleCycle);

            Vector3 newPosition = currentParticlePosition.Plus(positionDelta);
            Vector3 newVelocity = currentParticleVelocity.Plus(velocityDelta);

            // reflect off bowl
            float bowlZComponent = -((newPosition.Z * 0.4f) * (newPosition.Z * 0.4f));
            float bowlYComponent = -((newPosition.Y * 0.4f) * (newPosition.Y * 0.4f));
            float bowlX = bowlZComponent + bowlYComponent + 2.0f;

            // we're out side the bottom of the bowl, reflect
            float bowlReflectionCoeff = 0.5f;
            if(newPosition.X > bowlX)
            {
                Vector3 normal = new Vector3(newPosition.X, -0.32f * newPosition.Y, -0.32f * newPosition.Z).Normalized();

                newVelocity = newVelocity.Multiply(-1.0f);
                newVelocity = newVelocity.Minus(normal.Multiply((1.0f + bowlReflectionCoeff) * newVelocity.Dot(normal)));
                newPosition.X = bowlX;
            }

            // now the knife
            float knifeZComponent = -(((newPosition.Z + GMath.Sin(currentTime)) * 4.4f) * ((newPosition.Z + GMath.Sin(currentTime)) * 4.4f));
            float knifeYComponent = -(((newPosition.Y + GMath.Cos(currentTime)) * 4.4f) * ((newPosition.Y + GMath.Cos(currentTime)) * 4.4f));
            float knifeX = knifeZComponent + knifeYComponent + KnifeXOffsetAtTime(currentTime);

            // we're inside the knife, reflect
            float knifeReflectionCoeff = 1.0f;
            if (newPosition.X < knifeX)
            {
                Vector3 normal = new Vector3(newPosition.X, 
                    38.72f * (newPosition.Y + GMath.Cos(currentTime)), 
                    38.72f * (newPosition.Z + GMath.Sin(currentTime))).Normalized();

                //newVelocity = newVelocity.Multiply(-1.0f);
                //newVelocity = newVelocity.Plus(normal.Multiply((1.0f + knifeReflectionCoeff) * newVelocity.Dot(normal)));

                Vector3 knifeDelta = new Vector3(
                    KnifeXOffsetAtTime(currentTime) - KnifeXOffsetAtTime(currentTime - deltaTimePerParticleCycle),
                    GMath.Cos(currentTime) - GMath.Cos(currentTime - deltaTimePerParticleCycle),
                    GMath.Sin(currentTime) - GMath.Sin(currentTime - deltaTimePerParticleCycle));
                Vector3 knifeVelocity = knifeDelta.Divide(deltaTimePerParticleCycle);


                knifeVelocity.X = 0.0f;
                knifeVelocity = knifeVelocity.Multiply(0.25f); //.PiecewiseMultiply(normal);
                newVelocity = newVelocity.Minus(knifeVelocity);


                //knifeDelta.X = 0.0f;
                newPosition = newPosition.Minus(knifeDelta);

                //newPosition.X = knifeX; // moves down to tip
            }

            pXnew[particleIndex] = newPosition.X;
            pYnew[particleIndex] = newPosition.Y;
            pZnew[particleIndex] = newPosition.Z;

            vX[particleIndex] = newVelocity.X;
            vY[particleIndex] = newVelocity.Y;
            vZ[particleIndex] = newVelocity.Z;
        }

        [Cudafy]
        private static float SolveForSphere(Vector3 rayStart, Vector3 unitRay, Vector3 sphere, float radius)
        {
            Vector3 L = rayStart.Minus(sphere);
            float a = unitRay.Dot(unitRay);
            float b = 2.0f * unitRay.Dot(L);
            float c = L.Dot(L) - (radius * radius);

            float discriminant = (b * b) - (4.0f * a * c);

            if(discriminant <= 0.0f)
            {
                return float.NaN;
            }

            float t0 = (-b - GMath.Sqrt(discriminant)) / (2.0f * a);
            float t1 = (-b + GMath.Sqrt(discriminant)) / (2.0f * a);
            return GMath.Min(t0, t1);
        }

        [Cudafy]
        private static float SolveForParabaloidKnife(Vector3 rayStart, Vector3 unitRay, float currentTime)
        {
            //Vector3 squaredCoeffs = new Vector3(0.0f, 19.36f, 19.36f);
            //Vector3 linearCoeffs = new Vector3(1.0f, 0.0f, 0.0f);
            //float constantOffset = -1.8f;
            Vector3 squaredCoeffs = new Vector3(0.0f, knifeRenderWidthSquared, knifeRenderWidthSquared);
            Vector3 linearCoeffs = new Vector3(1.0f, 2.0f * knifeRenderWidthSquared * GMath.Cos(currentTime), 2.0f * knifeRenderWidthSquared * GMath.Sin(currentTime));
            float constantOffset = knifeRenderWidthSquared - (KnifeXOffsetAtTime(currentTime) + knifeRenderVerticalOffsetFromTime);

            float a = squaredCoeffs.X * unitRay.X * unitRay.X + 
                      squaredCoeffs.Y * unitRay.Y * unitRay.Y + 
                      squaredCoeffs.Z * unitRay.Z * unitRay.Z;
            float b = 2.0f * (squaredCoeffs.X * rayStart.X * unitRay.X + 
                              squaredCoeffs.Y * rayStart.Y * unitRay.Y + 
                              squaredCoeffs.Z * rayStart.Z * unitRay.Z) + linearCoeffs.Dot(unitRay);
            float c = squaredCoeffs.X * rayStart.X * rayStart.X + 
                      squaredCoeffs.Y * rayStart.Y * rayStart.Y + 
                      squaredCoeffs.Z * rayStart.Z * rayStart.Z + linearCoeffs.Dot(rayStart) + constantOffset;

            float discriminant = (b * b) - (4.0f * a * c);

            if (discriminant <= 0.0f)
            {
                return float.NaN;
            }

            float t0 = (-b - GMath.Sqrt(discriminant)) / (2.0f * a);
            float t1 = (-b + GMath.Sqrt(discriminant)) / (2.0f * a);
            return GMath.Min(t0, t1);
        }

        [Cudafy]
        public static void renderKernel(GThread thread, uint[,] image, float[] pX, float[] pY, float[] pZ, float[] colorPosition, float[] currentTimeArray)
        {
            int x = (thread.blockIdx.x * thread.blockDim.x) + thread.threadIdx.x;
            int y = (thread.blockIdx.y * thread.blockDim.y) + thread.threadIdx.y;

            float currentTime = currentTimeArray[0];

            Vector3 camera = new Vector3(-5, 8, -18); // was -1,8,18
            Vector3 frameTopLeft = new Vector3(-3, 3, 0);
            float frameWidth = 6.0f;
            float frameHeight = 6.0f;
            float radius = 0.04f;

            Vector3 rayThrough = new Vector3(frameTopLeft.X + frameWidth / width * x, frameTopLeft.Y - frameHeight / height * y, 0.0f);
            Vector3 initialRay = rayThrough.Minus(camera).Normalized();
            Vector3 lightPosition = new Vector3(-5.0f, -5.0f, -5.0f);
            bool hitBoundingSphere = false;
            float bestT = 100.0f; // far viewing plane
            int bestI = -1;

            for(int i = 0; i < pX.Length; i++)
            {
                Vector3 centerOfSphere = new Vector3(pX[i], pY[i], pZ[i]);

                float t = SolveForSphere(camera, initialRay, centerOfSphere, radius);
                
                if(!float.IsNaN(t))
                {
                    if(t < bestT)
                    {
                        bestT = t;
                        bestI = i;
                        hitBoundingSphere = true;
                    }
                }
            }

            float knifeT = SolveForParabaloidKnife(camera, initialRay, currentTime);
            bool hitKnife = false;
            if (!float.IsNaN(knifeT) && knifeT < bestT)
            {
                bestT = knifeT;
                hitKnife = true;
                hitBoundingSphere = false;
            }

            if (hitBoundingSphere || hitKnife)
            {
                Vector3 startOfInching = camera.Plus(initialRay.Multiply(bestT));
                float inchDelta = 0.001f;
                float totalInching = 0.0f;
                float maxInching = 0.5f;
                float threshold = 0.08f;
                bool hit = true; // false;
                Vector3 normal = new Vector3(0, 0, 0);
                Vector3 hitPosition = new Vector3(0, 0, 0);
                Vector3 contributingColor = new Vector3(0, 0, 0);

                if (hitBoundingSphere)
                {
                    Vector3 contributingSphere = new Vector3(pX[bestI], pY[bestI], pZ[bestI]);

                    Vector3 color1 = new Vector3(color1R, color1G, color1B);
                    Vector3 color2 = new Vector3(color2R, color2G, color2B);
                    float colorT = GMath.Min(GMath.Max(colorPosition[bestI], 0.0f), 1.0f);

                    Vector3 fromSphere = startOfInching.Minus(contributingSphere);

                    normal = fromSphere.Normalized();  // new Vector3(0, 0, 0);
                    hitPosition = startOfInching;
                    contributingColor = color1.ColorBlend(color2, colorT);
                }
                else if(hitKnife)
                {
                    hitPosition = startOfInching;
                    normal = new Vector3(hitPosition.X,
                    2.0f * knifeRenderWidthSquared * (hitPosition.Y + GMath.Cos(currentTime)),
                    2.0f * knifeRenderWidthSquared * (hitPosition.Z + GMath.Sin(currentTime))).Normalized();

                    contributingColor = new Vector3(0.8f, 0.7f, 0.7f);
                }


                //Vector3 currentPosition = new Vector3(0, 0, 0);

                // iso sphere's not today
                //while (totalInching < maxInching)
                //{
                //    currentPosition = startOfInching.Plus(initialRay.Multiply(totalInching));
                //    totalInching += inchDelta;

                //    // the contributions from each sphere
                //    float sumContributions = 0.0f;
                //    Vector3 contributedNormals = new Vector3(0, 0, 0);

                //    for (int i = 0; i < pX.Length; i++)
                //    {
                //        Vector3 contributingSphere = new Vector3(pX[i], pY[i], pZ[i]);
                //        Vector3 fromSphere = currentPosition.Minus(contributingSphere);
                //        float distanceToContributor = fromSphere.Magnitude();

                //        if (distanceToContributor < 0.03535f)
                //        {
                //            float functionX = distanceToContributor * 20.0f;
                //            float contribution = ((functionX * functionX * functionX * functionX) -
                //                (functionX * functionX) + 0.25f);
                //            //float contribution = 1.0f / (distanceToContributor * distanceToContributor);

                //            sumContributions += contribution;
                //            contributedNormals = contributedNormals.Plus(fromSphere.Normalized().Multiply(contribution));
                //        }
                //    }

                //    if (sumContributions >= threshold)
                //    {
                //        hit = true;
                //        normal = contributedNormals.Normalized();
                //        break;
                //    }
                //}

                if (hit)
                {
                    float specularHighlightHardness = 10.0f;

                    // normalized vector from hit position to light position
                    Vector3 hitToLight = lightPosition.Minus(hitPosition).Normalized();

                    // normalized vector from hit position to eye/camera
                    Vector3 hitToEye = camera.Minus(hitPosition).Normalized();
                    // H in specular component
                    Vector3 eyeHalfVector = hitToEye.Plus(hitToLight).Normalized();

                    // diffuse factor
                    float diffuseFactor = GMath.Max(hitToLight.Dot(normal), 0.0f);
                    float specularFactor = GMath.Pow(GMath.Max(eyeHalfVector.Dot(normal), 0.0f), specularHighlightHardness);

                    // light coefficients
                    float kspecular = 0.4f;
                    float kdiffuse = 0.5f;
                    float kambient = 0.1f;

                    // color
                    Vector3 color = contributingColor;

                    // final color
                    Vector3 ambient = color.Multiply(kambient);
                    Vector3 diffuse = color.Multiply(diffuseFactor * kdiffuse);
                    Vector3 specular = color.Multiply(specularFactor * kspecular);

                    Vector3 finalColor = ambient.Plus(diffuse).Plus(specular);

                    uint shiftr = (uint)(finalColor.X * 255.0f);
                    uint shiftg = (uint)(finalColor.Y * 255.0f);
                    uint shiftb = (uint)(finalColor.Z * 255.0f);

                    image[x, y] = (shiftr << 16) | (shiftg << 8) | (shiftb << 0);
                }
                else
                {
                    image[x, y] = 0x00FFFF00;
                }
            }
                
            else
            {
                image[x, y] = 0x00FFFF00;
            }
        }

        #region Vector3

        [Cudafy]
        struct Vector3
        {
            public float X, Y, Z;

            public Vector3(float x, float y, float z)
            {
                X = x; Y = y; Z = z;
            }

            public Vector3 Minus(Vector3 v2)
            {
                return new Vector3(
                    this.X - v2.X,
                    this.Y - v2.Y,
                    this.Z - v2.Z);
            }

            public Vector3 Plus(Vector3 v2)
            {
                return new Vector3(
                    this.X + v2.X,
                    this.Y + v2.Y,
                    this.Z + v2.Z);
            }

            public float Magnitude()
            {
                float sum = X * X + Y * Y + Z * Z;

                if (sum == 0.0f)
                {
                    return 0.0f;
                }
                    
                return GMath.Sqrt(X * X + Y * Y + Z * Z);
            }

            public Vector3 Normalized()
            {
                float magnitude = Magnitude();

                if(magnitude == 0.0f)
                {
                    return new Vector3(0.0f, 0.0f, 0.0f);
                }

                return new Vector3(X / magnitude, Y / magnitude, Z / magnitude);
            }

            public Vector3 Multiply(float factor)
            {
                return new Vector3(X * factor, Y * factor, Z * factor);
            }

            public Vector3 PiecewiseMultiply(Vector3 other)
            {
                return new Vector3(
                    this.X * other.X,
                    this.Y * other.Y,
                    this.Z * other.Z);
            }

            public Vector3 Divide(float denomonator)
            {
                return new Vector3(X / denomonator, Y / denomonator, Z / denomonator);
            }

            public float Dot(Vector3 other)
            {
                return X * other.X + Y * other.Y + Z * other.Z;
            }

            public Vector3 ColorBlend(Vector3 other, float t)
            {
                return new Vector3(
                    GMath.Sqrt((1.0f - t) * X * X + t * other.X * other.X),
                    GMath.Sqrt((1.0f - t) * Y * Y + t * other.Y * other.Y),
                    GMath.Sqrt((1.0f - t) * Z * Z + t * other.Z * other.Z)
                );
            }
        }

        #endregion
    }
}
